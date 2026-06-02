import bpy
from bpy.props import (
    StringProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
    PointerProperty,
)
from common import DEPTHS_DIR
from render import render_rgb, render_depth
from export_scene import ExportSceneConfig, export_scene, generate_all
from spawn_cameras import clear_cameras, apply_lookat, spawn_cameras, SpawnCamerasConfig

bl_info = {
    "name": "COLDER - Synthetic SfM Dataset Generator in COLMAP format",
    "author": "Mateo de Mayo",
    "blender": (5, 1, 1),
    "description": "Tools to help create synthetic SfM data and export it to COLMAP format",
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

    generate_rgb: BoolProperty(name="Generate RGB Renders", default=esc.GENERATE_RGB)
    generate_depths: BoolProperty(name="Generate Depth Renders", default=esc.GENERATE_DEPTHS)
    generate_debug_depths: BoolProperty(name="Generate Debug Depth Maps", default=esc.GENERATE_DEBUG_DEPTHS)
    depth_occlusion: BoolProperty(
        name="Depthmaps Occlusions",
        default=esc.DEPTH_OCCLUSION,
        description=f"Will use depthmaps from 'EXPORT_PATH/sparse/0/{DEPTHS_DIR}' folder",
    )
    depth_occlusion_thresh: FloatProperty(
        name="Depth Occlusion Thresh (m)", default=esc.DEPTH_OCCLUSION_THRESH, min=0.0
    )

    generate_colmap: BoolProperty(name="Generate COLMAP Scene", default=True)

    # Camera spawn
    # defcurves = spc.BEZIER_CURVE_LIST and ",".join(spc.BEZIER_CURVE_LIST) or ""
    defcurves = ""
    bezier_curve_list: StringProperty(name="Curves", description="comma list of names (empty=all)", default=defcurves)
    lookup_target: StringProperty(name="Look-at Target", default=spc.LOOKUP_TARGET)

    number_of_cameras: IntProperty(name="Number of Cameras", default=spc.NUMBER_OF_CAMERAS, min=1)
    samples_per_bezier_segment: IntProperty(name="Bezier Samples", default=spc.SAMPLES_PER_BEZIER_SEGMENT, min=4)

    # Collection handling
    # TODO@mateosss: Of course I want to use a camera collection, remove this option
    use_collection: BoolProperty(name="Use Camera Collection", default=True)
    camera_collection_name: StringProperty(name="Camera Collection Name", default="SpawnedCameras")


# ------------------------------------------------------------------------
# Operators
# ------------------------------------------------------------------------


def make_export_config(context: bpy.types.Context) -> ExportSceneConfig:
    return ExportSceneConfig(
        EXPORT_PATH=context.scene.colder_props.export_path,
        TARGET_OBJECTS=[n.strip() for n in context.scene.colder_props.target_objects.split(",") if n.strip()],
        POINT_3D_SAVE_NOISE_STDEV=context.scene.colder_props.point_3d_noise,
        POINT_2D_SAVE_NOISE_STDEV=context.scene.colder_props.point_2d_noise,
        POSE_TRANSLATION_NOISE_STDEV=context.scene.colder_props.pose_translation_noise,
        POSE_ROTATION_NOISE_STDEV=context.scene.colder_props.pose_rotation_noise,
        POINT_3D_DENSITY=context.scene.colder_props.point_3d_density,
        POINT_2D_DENSITY=context.scene.colder_props.point_2d_density,
        MIN_NUM_OBS_PER_POINT3D=context.scene.colder_props.min_num_obs_per_point3d,
        GENERATE_RGB=context.scene.colder_props.generate_rgb,
        GENERATE_DEPTHS=context.scene.colder_props.generate_depths,
        GENERATE_DEBUG_DEPTHS=context.scene.colder_props.generate_debug_depths,
        DEPTH_OCCLUSION=context.scene.colder_props.depth_occlusion,
        DEPTH_OCCLUSION_THRESH=context.scene.colder_props.depth_occlusion_thresh,
        GENERATE_COLMAP=context.scene.colder_props.generate_colmap,
        IMAGE_NAME_FMT=context.scene.colder_props.image_name_fmt,
    )


def make_camera_spawn_config(context: bpy.types.Context):
    return SpawnCamerasConfig(
        BEZIER_CURVE_LIST=[n.strip() for n in context.scene.colder_props.bezier_curve_list.split(",") if n.strip()],
        LOOKUP_TARGET=context.scene.colder_props.lookup_target,
        NUMBER_OF_CAMERAS=context.scene.colder_props.number_of_cameras,
        SAMPLES_PER_BEZIER_SEGMENT=context.scene.colder_props.samples_per_bezier_segment,
        USE_COLLECTION=context.scene.colder_props.use_collection,
        CAMERA_COLLECTION_NAME=context.scene.colder_props.camera_collection_name,
    )


class COLDER_OT_spawn_cameras(bpy.types.Operator):
    bl_idname = "colder.spawn_cameras"
    bl_label = "Spawn Cameras"

    def execute(self, context):
        try:
            config = make_camera_spawn_config(context)
            spawn_cameras(config)
        except (RuntimeError, ValueError) as e:
            self.report({"ERROR"}, f"Error spawning cameras: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Spawned {config.NUMBER_OF_CAMERAS} cameras")
        return {"FINISHED"}


class COLDER_OT_export_scene(bpy.types.Operator):
    bl_idname = "colder.export_scene"
    bl_label = "Export Scene"

    def execute(self, context):
        try:
            config = make_export_config(context)
            export_scene(config)
            self.report({"INFO"}, f"Scene exported to: {config.EXPORT_PATH}")
        except (RuntimeError, ValueError) as e:
            self.report({"ERROR"}, f"Error exporting scene: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


class COLDER_OT_render_depthmaps(bpy.types.Operator):
    bl_idname = "colder.render_depthmaps"
    bl_label = "Render Depthmaps"

    def execute(self, context):
        try:
            export_path = context.scene.colder_props.export_path
            output_dir = render_depth(export_path, render_dbg=context.scene.colder_props.generate_debug_depths)
        except (RuntimeError, ValueError) as e:
            self.report({"ERROR"}, f"Error rendering depthmaps: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Depthmaps written to: {output_dir}")
        return {"FINISHED"}


class COLDER_OT_render_images(bpy.types.Operator):
    bl_idname = "colder.render_images"
    bl_label = "Render Images"

    def execute(self, context):
        try:
            export_path = context.scene.colder_props.export_path
            output_dir = render_rgb(export_path)
        except (RuntimeError, ValueError) as e:
            self.report({"ERROR"}, f"Error rendering images: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Images written to: {output_dir}")
        return {"FINISHED"}


class COLDER_OT_generate_all(bpy.types.Operator):
    bl_idname = "colder.generate_all"
    bl_label = "Generate All"

    def execute(self, context):
        try:
            config = make_export_config(context)
            generate_all(config)
        except (RuntimeError, ValueError) as e:
            self.report({"ERROR"}, f"Error generating all data: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"All data generated in: {config.EXPORT_PATH}")
        return {"FINISHED"}


class COLDER_OT_clear_cameras(bpy.types.Operator):
    bl_idname = "colder.clear_cameras"
    bl_label = "Clear Cameras"

    def execute(self, context):
        try:
            clear_cameras()
        except (RuntimeError, ValueError) as e:
            self.report({"ERROR"}, f"Error clearing cameras: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Cameras cleared")
        return {"FINISHED"}


class COLDER_OT_apply_lookat(bpy.types.Operator):
    bl_idname = "colder.apply_lookat"
    bl_label = "Apply Look-At"

    def execute(self, context):
        try:
            apply_lookat()
        except (RuntimeError, ValueError) as e:
            self.report({"ERROR"}, f"Error applying look-at constraints: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Look-at constraints applied")
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

        layout.label(text="Generation")
        layout.prop(props, "export_path")

        box = layout.box()
        box.prop(props, "generate_rgb")
        if props.generate_rgb:
            box.operator("colder.render_images")
            # TODO: add option for rendered point size

        box = layout.box()
        box.prop(props, "generate_depths")
        if props.generate_depths:
            box.prop(props, "generate_debug_depths")
            box.operator("colder.render_depthmaps")

        box = layout.box()
        box.prop(props, "generate_colmap")
        if props.generate_colmap:
            box.prop(props, "target_objects")
            box.prop(props, "depth_occlusion")
            box.prop(props, "depth_occlusion_thresh")
            box.operator("colder.export_scene")

        layout.operator("colder.generate_all")

        layout.separator()


# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (
    COLDER_Properties,
    COLDER_OT_spawn_cameras,
    COLDER_OT_export_scene,
    COLDER_OT_generate_all,
    COLDER_OT_clear_cameras,
    COLDER_OT_apply_lookat,
    COLDER_OT_render_depthmaps,
    COLDER_OT_render_images,
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
