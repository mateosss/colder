import sys
import importlib
from pathlib import Path
import re
import bpy
from bpy.props import (
    StringProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
    PointerProperty,
)


bl_info = {
    "name": "COLDER - Colmap Export Helper",
    "author": "Mateo de Mayo",
    "blender": (4, 0, 0),
    "description": "Tools to help export Blender scenes and spawn cameras in COLMAP format",
    "location": "View3D > Sidebar > COLDER",
    "category": "3D View",
}


# ------------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------------


class COLDER_Properties(bpy.types.PropertyGroup):
    from spawn_cameras import SpawnCamerasConfig as spc
    from export_scene import ExportSceneConfig as esc

    # Export options
    export_path: StringProperty(name="Export Path", default=esc.EXPORT_PATH, subtype="DIR_PATH")
    # defobj = esc.TARGET_OBJECTS and ",".join(esc.TARGET_OBJECTS) or ""
    defobj = ""  # default to empty = all objects
    target_objects: StringProperty(name="Target Objects", description="comma list of names (empty=all)", default=defobj)
    image_name_fmt: StringProperty(name="Image Name Format", default=esc.IMAGE_NAME_FMT)

    # Noise
    point_3d_noise: FloatProperty(name="3D Point Noise (m)", default=esc.POINT_3D_SAVE_NOISE_STDEV, min=0.0)
    point_2d_noise: FloatProperty(name="2D Point Noise (px)", default=esc.POINT_2D_SAVE_NOISE_STDEV, min=0.0)
    pose_translation_noise: FloatProperty(
        name="Pose Translation Noise (m)", default=esc.POSE_TRANSLATION_NOISE_STDEV, min=0.0
    )
    pose_rotation_noise: FloatProperty(name="Pose Rotation Noise (deg)", default=esc.POSE_ROTATION_NOISE_STDEV, min=0.0)

    # Density / filtering
    point_3d_density: FloatProperty(
        name="3D Point Density", default=esc.POINT_3D_DENSITY, min=0.0, max=1.0, subtype="FACTOR"
    )
    point_2d_density: FloatProperty(
        name="2D Observation Density", default=esc.POINT_2D_DENSITY, min=0.0, max=1.0, subtype="FACTOR"
    )
    min_num_obs_per_point3d: IntProperty(name="Min Observations / 3D Point", default=esc.MIN_NUM_OBS_PER_POINT3D, min=1)
    depthmaps_occlusions: BoolProperty(name="Depthmaps Occlusions", default=esc.DEPTHMAPS_OCCLUSIONS)
    depth_occlusion_thresh: FloatProperty(
        name="Depth Occlusion Thresh (m)",
        default=esc.DEPTH_OCCLUSION_THRESH,
        min=0.0,
    )

    # Camera spawn
    # defcurves = spc.BEZIER_CURVE_LIST and ",".join(spc.BEZIER_CURVE_LIST) or ""
    defcurves = ""
    bezier_curve_list: StringProperty(name="Curves", description="comma list of names (empty=all)", default=defcurves)
    lookup_target: StringProperty(name="Look-at Target", default=spc.LOOKUP_TARGET)

    number_of_cameras: IntProperty(name="Number of Cameras", default=spc.NUMBER_OF_CAMERAS, min=1)
    samples_per_bezier_segment: IntProperty(name="Bezier Samples", default=spc.SAMPLES_PER_BEZIER_SEGMENT, min=4)

    # Collection handling
    use_collection: BoolProperty(name="Use Camera Collection", default=True)
    camera_collection_name: StringProperty(name="Camera Collection Name", default="SpawnedCameras")


# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------


def ensure_blend_dir_on_syspath():
    # Directory of the currently saved .blend
    if not bpy.data.filepath:
        return None  # unsaved file
    script_dir = str(Path(bpy.data.filepath).parent)
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    return script_dir


def run_module_main(module_name: str):
    script_dir = ensure_blend_dir_on_syspath()
    if script_dir is None:
        raise RuntimeError("Save the .blend file first (needed to locate external scripts).")

    mod = importlib.import_module(module_name)
    importlib.reload(mod)  # so edits are picked up without restarting Blender
    if not hasattr(mod, "main"):
        raise RuntimeError(f"Module '{module_name}.py' has no main()")
    mod.main()


def _safe_stem(name: str) -> str:
    # Keep filenames portable and deterministic.
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _extract_z_to_npz(exr_path: str, npz_path: str):
    """Extract Z buffer from EXR using Blender's image loading and save as NPZ."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Depthmaps requires numpy in Blender's Python environment") from exc

    # Load EXR into Blender's image cache
    img = bpy.data.images.load(exr_path, check_existing=False)
    try:
        # Extract pixel data as numpy array
        pixels = np.array(img.pixels[:])
        width = img.size[0]
        height = img.size[1]
        channels = len(img.pixels) // (width * height)

        # Reshape to (height, width, channels) then extract Z (usually last channel for depth)
        rgba = pixels.reshape((height, width, channels))

        # For OPEN_EXR with Z pass, the Z is typically written to one of the color channels
        # or as a separate layer. We'll extract the first channel as depth approximation.
        depth = rgba[:, :, 0].astype(np.float32)

        # Save as NPZ
        np.savez_compressed(npz_path, depth=depth)
    finally:
        bpy.data.images.remove(img)


def generate_depthmaps(export_path: str):
    scene = bpy.context.scene
    cameras = sorted((obj for obj in scene.objects if obj.type == "CAMERA"), key=lambda obj: obj.name)
    if not cameras:
        raise RuntimeError("No camera objects found in the scene")

    depth_dir = Path(bpy.path.abspath(export_path)) / "depthmaps"
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Keep current render config untouched after batch render.
    orig_camera = scene.camera
    orig_filepath = scene.render.filepath
    orig_format = scene.render.image_settings.file_format
    orig_color_mode = scene.render.image_settings.color_mode
    orig_color_depth = scene.render.image_settings.color_depth
    orig_exr_codec = scene.render.image_settings.exr_codec
    orig_use_zbuffer = getattr(scene.render.image_settings, "use_zbuffer", None)
    orig_resolution_x = scene.render.resolution_x
    orig_resolution_y = scene.render.resolution_y
    orig_view_layer_depth = [vl.use_pass_z for vl in scene.view_layers]

    try:
        for view_layer in scene.view_layers:
            view_layer.use_pass_z = True

        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "ZIP"
        if hasattr(scene.render.image_settings, "use_zbuffer"):
            scene.render.image_settings.use_zbuffer = True

        for cam in cameras:
            scene.camera = cam

            if "width" in cam.data and "height" in cam.data:
                scene.render.resolution_x = int(cam.data["width"])
                scene.render.resolution_y = int(cam.data["height"])

            # Render to temporary EXR
            temp_exr = depth_dir / f"{_safe_stem(cam.name)}_temp.exr"
            scene.render.filepath = str(temp_exr)
            bpy.ops.render.render(write_still=True)

            # Extract Z from EXR and save as NPZ using Blender's image API
            _extract_z_to_npz(str(temp_exr), str(depth_dir / f"{_safe_stem(cam.name)}.npz"))
            temp_exr.unlink()
    finally:
        scene.camera = orig_camera
        scene.render.filepath = orig_filepath
        scene.render.image_settings.file_format = orig_format
        scene.render.image_settings.color_mode = orig_color_mode
        scene.render.image_settings.color_depth = orig_color_depth
        scene.render.image_settings.exr_codec = orig_exr_codec
        if orig_use_zbuffer is not None:
            scene.render.image_settings.use_zbuffer = orig_use_zbuffer
        scene.render.resolution_x = orig_resolution_x
        scene.render.resolution_y = orig_resolution_y
        for view_layer, use_pass_z in zip(scene.view_layers, orig_view_layer_depth):
            view_layer.use_pass_z = use_pass_z


# ------------------------------------------------------------------------
# Operators
# ------------------------------------------------------------------------


class COLDER_OT_spawn_cameras(bpy.types.Operator):
    bl_idname = "colder.spawn_cameras"
    bl_label = "Spawn Cameras"

    def execute(self, context):
        from spawn_cameras import SpawnCamerasConfig, spawn_cameras

        config = SpawnCamerasConfig(
            BEZIER_CURVE_LIST=[n.strip() for n in context.scene.colder_props.bezier_curve_list.split(",") if n.strip()],
            LOOKUP_TARGET=context.scene.colder_props.lookup_target,
            NUMBER_OF_CAMERAS=context.scene.colder_props.number_of_cameras,
            SAMPLES_PER_BEZIER_SEGMENT=context.scene.colder_props.samples_per_bezier_segment,
            USE_COLLECTION=context.scene.colder_props.use_collection,
            CAMERA_COLLECTION_NAME=context.scene.colder_props.camera_collection_name,
        )
        spawn_cameras(config)
        return {"FINISHED"}


class COLDER_OT_export_scene(bpy.types.Operator):
    bl_idname = "colder.export_scene"
    bl_label = "Export Scene"

    def execute(self, context):
        from export_scene import ExportSceneConfig, export_scene

        config = ExportSceneConfig(
            EXPORT_PATH=context.scene.colder_props.export_path,
            TARGET_OBJECTS=[n.strip() for n in context.scene.colder_props.target_objects.split(",") if n.strip()],
            POINT_3D_SAVE_NOISE_STDEV=context.scene.colder_props.point_3d_noise,
            POINT_2D_SAVE_NOISE_STDEV=context.scene.colder_props.point_2d_noise,
            POSE_TRANSLATION_NOISE_STDEV=context.scene.colder_props.pose_translation_noise,
            POSE_ROTATION_NOISE_STDEV=context.scene.colder_props.pose_rotation_noise,
            POINT_3D_DENSITY=context.scene.colder_props.point_3d_density,
            POINT_2D_DENSITY=context.scene.colder_props.point_2d_density,
            MIN_NUM_OBS_PER_POINT3D=context.scene.colder_props.min_num_obs_per_point3d,
            DEPTHMAPS_OCCLUSIONS=context.scene.colder_props.depthmaps_occlusions,
            DEPTH_OCCLUSION_THRESH=context.scene.colder_props.depth_occlusion_thresh,
            IMAGE_NAME_FMT="cam_{:04d}.png",
        )
        export_scene(config)
        return {"FINISHED"}


class COLDER_OT_generate_depthmaps(bpy.types.Operator):
    bl_idname = "colder.generate_depthmaps"
    bl_label = "Generate Depthmaps"

    def execute(self, context):
        export_path = context.scene.colder_props.export_path
        try:
            generate_depthmaps(export_path)
        except (RuntimeError, ValueError) as exc:
            self.report({"ERROR"}, f"Depthmap generation failed: {exc}")
            return {"CANCELLED"}

        output_dir = Path(bpy.path.abspath(export_path)) / "depthmaps"
        self.report({"INFO"}, f"Depthmaps written to: {output_dir}")
        return {"FINISHED"}


class COLDER_OT_clear_cameras(bpy.types.Operator):
    bl_idname = "colder.clear_cameras"
    bl_label = "Clear Cameras"

    def execute(self, context):
        from spawn_cameras import clear_cameras
        clear_cameras()
        return {"FINISHED"}

class COLDER_OT_apply_lookat(bpy.types.Operator):
    bl_idname = "colder.apply_lookat"
    bl_label = "Apply Look-At"

    def execute(self, context):
        from spawn_cameras import apply_lookat
        apply_lookat()
        return {"FINISHED"}

# ------------------------------------------------------------------------
# UI Panel
# ------------------------------------------------------------------------


class COLDER_PT_panel(bpy.types.Panel):
    bl_idname = "COLDER_PT_panel"
    bl_label = "COLDER Tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "COLDER"  # N-panel tab name

    def draw(self, context):
        layout = self.layout
        props = context.scene.colder_props

        layout.label(text="1. Camera Spawn")
        layout.prop(props, "number_of_cameras")
        layout.prop(props, "lookup_target")
        layout.prop(props, "bezier_curve_list")
        layout.prop(props, "samples_per_bezier_segment")
        layout.prop(props, "use_collection")
        if props.use_collection:
            layout.prop(props, "camera_collection_name")
        layout.operator("colder.spawn_cameras")
        layout.operator("colder.generate_depthmaps")
        layout.operator("colder.clear_cameras")
        layout.operator("colder.apply_lookat")

        layout.separator()
        layout.label(text="2. Export Scene")

        layout.label(text="Noise")
        layout.prop(props, "point_3d_noise")
        layout.prop(props, "point_2d_noise")
        layout.prop(props, "pose_translation_noise")
        layout.prop(props, "pose_rotation_noise")
        layout.separator()

        layout.label(text="Density / Filtering")
        layout.prop(props, "point_3d_density")
        layout.prop(props, "point_2d_density")
        layout.prop(props, "min_num_obs_per_point3d")
        layout.prop(props, "depthmaps_occlusions")
        if props.depthmaps_occlusions:
            layout.prop(props, "depth_occlusion_thresh")
        layout.separator()

        layout.prop(props, "export_path")
        layout.prop(props, "target_objects")
        layout.operator("colder.export_scene")
        layout.separator()


# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (
    COLDER_Properties,
    COLDER_OT_spawn_cameras,
    COLDER_OT_generate_depthmaps,
    COLDER_OT_export_scene,
    COLDER_OT_clear_cameras,
    COLDER_OT_apply_lookat,
    COLDER_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.colder_props = PointerProperty(type=COLDER_Properties)


def unregister():
    del bpy.types.Scene.colder_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


def main():
    register()
