from tqdm import tqdm
import re
import bpy
from pathlib import Path
from typing import Optional
from common import DEPTHS_DIR, IMAGES_DIR, stdout_redirected
from PIL import Image
import shutil


def _safe_stem(name: str) -> str:
    # Keep filenames portable and deterministic.
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _extract_z_to_npz(exr_path: str, npz_path: str, png_path: Optional[str] = None):
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

        # Invert Y axis to match image coordinates
        depth = np.flipud(depth)

        # Save as NPZ
        np.savez_compressed(npz_path, depth=depth)

        if png_path is not None:
            # TODO@mateosss: have DMAX a variable somewhere
            DMAX = 110.0  # max depth for normalization (tune as needed)
            depth_normalized = np.clip(depth, 0, DMAX) / DMAX  # normalize to [0, 1]
            depth_rgba = np.stack([depth_normalized] * 3 + [np.ones_like(depth_normalized)], axis=-1)  # grayscale RGBA
            depth_rgba_uint8 = (depth_rgba * 255).astype(np.uint8)
            Image.fromarray(depth_rgba_uint8).save(png_path)
    finally:
        bpy.data.images.remove(img)


def render_(export_path: Path, rtype: str = "DEPTH", render_depth_dbg: bool = False) -> Path:
    if rtype == "DEPTH":
        file_format = "OPEN_EXR"
        color_depth = "32"
        ext = "exr"
        path = DEPTHS_DIR
    elif rtype == "COLOR":
        file_format = "PNG"
        color_depth = "8"
        ext = "png"
        path = IMAGES_DIR
    else:
        raise ValueError(f"Unsupported render type: {rtype}")

    scene = bpy.context.scene
    # TODO@mateosss: use regex match cam_000X to be more robust to naming
    cameras = sorted(
        (obj for obj in scene.objects if obj.type == "CAMERA" and re.match(r"cam_\d{4}", obj.name)),
        key=lambda obj: obj.name,
    )
    if not cameras:
        raise RuntimeError("No camera objects found in the scene")

    path = Path(export_path) / path
    if path.exists():
        print(f"Warning: {rtype} {path=} already exists, deleting")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

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

    assert len(scene.view_layers) == 1, "We expect a unique view layer"
    view_layer = scene.view_layers[0]

    try:
        view_layer.use_pass_z = True

        scene.render.image_settings.file_format = file_format
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = color_depth
        scene.render.image_settings.exr_codec = "ZIP"
        if hasattr(scene.render.image_settings, "use_zbuffer"):
            scene.render.image_settings.use_zbuffer = True

        # TODO@mateosss: Delete previous images in folder
        for cam in tqdm(cameras, desc=f"Rendering {rtype}"):
            scene.camera = cam
            scene.camera.data.sensor_fit = "HORIZONTAL"
            scene.camera.data.sensor_width = 36.0

            assert "width" in cam.data and "height" in cam.data, f"Is {cam=} a properly spawned camera?"

            scene.render.resolution_x = int(cam.data["width"])
            scene.render.resolution_y = int(cam.data["height"])

            # Render to temporary EXR
            # temp_img = img_dir / f"{_safe_stem(cam.name)}.{ext}"
            # TODO@mateosss: Fix naming, expect cam_000X
            # cam_id = int(cam.name[3:6]) + 1
            cam_id = int(cam.name.split("_")[1])
            name: str = f"cam_{cam_id:04d}.{ext}"
            temp_img = path / f"{name}"

            scene.render.filepath = str(temp_img)
            with stdout_redirected():  # suppress render log spam
                bpy.ops.render.render(write_still=True)

            # Extract Z from EXR and save as NPZ using Blender's image API
            if rtype == "DEPTH":
                png = None
                if render_depth_dbg:
                    png_dir = path.with_suffix(".debug")
                    png_dir.mkdir(parents=True, exist_ok=True)
                    png = str(png_dir / f"{_safe_stem(cam.name)}.png")
                npz = str(path / f"{_safe_stem(cam.name)}.npz")
                _extract_z_to_npz(str(temp_img), npz, png)
                temp_img.unlink()
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

    return path


def prepare_render(rtype: str = "DEPTH"):
    # TODO@mateosss: These should all be configurable somewhere
    target_objects = ["Courthouse"]

    RENDER_MOD = "POINT_CLOUD_RENDER"
    MESH_MOD = "POINT_CLOUD_MESH"
    if rtype == "DEPTH":
        props = {"Threshold": 0.2}  # TODO@mateosss: make this a variable somewhere
        onmod = MESH_MOD
        offmod = RENDER_MOD
    elif rtype == "COLOR":
        props = {"Radius": 0.15}  # TODO@mateosss: make this a variable somewhere
        onmod = RENDER_MOD
        offmod = MESH_MOD
    else:
        raise ValueError(f"Unsupported render type: {rtype}")

    for obj_name in target_objects:
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue

        # Disable the other modifier
        offmod = obj.modifiers[offmod]
        offmod.show_viewport = False
        offmod.show_render = False

        # Enable target modifier
        onmod = obj.modifiers[onmod]
        for item in onmod.node_group.interface.items_tree:
            if item.name in props:
                onmod[item.identifier] = props[item.name]
        onmod.show_viewport = True
        onmod.show_render = True


def render_rgb(export_path: Path) -> Path:
    prepare_render("COLOR")
    return render_(export_path, rtype="COLOR")


def render_depth(export_path: Path, render_dbg: bool = False) -> Path:
    prepare_render("DEPTH")
    return render_(export_path, rtype="DEPTH", render_depth_dbg=render_dbg)
