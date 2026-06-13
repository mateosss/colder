# Blender 5.0.1
# Spawns cameras along one or more Bezier curve objects, evenly spaced by arc length,
# and makes each camera look at a target object.
#
# Usage:
# - Put this in Blender's Text Editor and Run Script.
# - Ensure the curve objects and target object exist in the scene.

import bpy
from mathutils import Vector, Matrix
from mathutils.geometry import interpolate_bezier  # available in Blender's mathutils
from dataclasses import dataclass, field


@dataclass
class SpawnCamerasConfig:
    BEZIER_CURVE_LIST: list[str] = field(default_factory=list)  # names of curve objects
    LOOKUP_TARGET: str = "Empty"  # object name all cameras will look at
    NUMBER_OF_CAMERAS: int = 10

    # Sampling density for arc-length approximation (higher = more accurate, slower)
    SAMPLES_PER_BEZIER_SEGMENT: int = 64

    # If True, creates/uses a collection to keep things tidy
    USE_COLLECTION: bool = True
    CAMERA_COLLECTION_NAME: str = "SpawnedCameras"

    ANIMATION_CAMERA: str = "Camera"  # Main camera with animation to use
    ANIMATION_STEP: int = 10  # How many frames to skip


def get_initial_intrinsics(_: int) -> dict:
    # parameter is intentionally unused for now; keep it for future per-camera variation.
    return {
        "sensor_width_m": 0.036,  # 36mm is blender's default
        "model": "SIMPLE_RADIAL",
        "width": 640,
        "height": 480,
        "params": [480, 640 / 2, 480 / 2, 0.0],  # f (f=fx=fy), cx, cy, k1
    }


def _ensure_collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _evaluated_curve_world(obj_curve: bpy.types.Object):
    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj_curve.evaluated_get(dg)
    curve_eval = obj_eval.data
    return obj_eval, curve_eval


def _sample_curve_world_points(obj_curve: bpy.types.Object, samples_per_segment: int):
    if obj_curve.type != "CURVE":
        raise TypeError(f"Object '{obj_curve.name}' is not a CURVE")

    obj_eval, curve_eval = _evaluated_curve_world(obj_curve)
    world = obj_eval.matrix_world

    all_pts_world = []

    for spline in curve_eval.splines:
        if spline.type != "BEZIER":
            continue

        bp = spline.bezier_points
        n = len(bp)
        if n < 2:
            continue

        # segment i goes from bp[i] to bp[i+1], plus cyclic last->first
        seg_count = n if spline.use_cyclic_u else (n - 1)

        for i in range(seg_count):
            a = bp[i]
            b = bp[(i + 1) % n]

            # interpolate_bezier returns a list of Vectors in local space
            pts = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, samples_per_segment + 1)

            # avoid duplicating the joint point between segments
            if all_pts_world:
                pts = pts[1:]

            for p in pts:
                all_pts_world.append(world @ Vector(p))

    if len(all_pts_world) < 2:
        raise RuntimeError(f"Curve '{obj_curve.name}' yielded too few sampled points (need at least 2).")

    return all_pts_world


def _arc_length_parameterization(points_world):
    # Returns cumulative lengths (same length as points_world) and total length.
    cum = [0.0]
    total = 0.0
    for i in range(1, len(points_world)):
        total += (points_world[i] - points_world[i - 1]).length
        cum.append(total)
    return cum, total


def _point_at_distance(points_world, cum_lengths, dist):
    # Linear interpolation between sampled points by arc length.
    if dist <= 0.0:
        return points_world[0].copy()
    if dist >= cum_lengths[-1]:
        return points_world[-1].copy()

    # binary search
    lo, hi = 0, len(cum_lengths) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cum_lengths[mid] < dist:
            lo = mid
        else:
            hi = mid

    d0 = cum_lengths[lo]
    d1 = cum_lengths[hi]
    t = 0.0 if d1 == d0 else (dist - d0) / (d1 - d0)
    return points_world[lo].lerp(points_world[hi], t)


def _create_camera(
    name: str,
    location: Vector,
    target_obj: bpy.types.Object,
    intrinsics: dict,
    collection=None,
):
    cam_data = bpy.data.cameras.new(name + "_DATA")
    # Set camera focal length and sensor size based on intrinsics (assuming SIMPLE_RADIAL)
    if intrinsics["model"] == "SIMPLE_RADIAL":
        f, cx, cy, k1 = intrinsics["params"]
        sensor_width_mm = intrinsics["sensor_width_m"] * 1000
        cols = intrinsics["width"]
        px_width = sensor_width_mm / cols
        fx = f  # f is fx=fy in SIMPLE_RADIAL, not focal length in mm
        focal_length_mm = fx * px_width
        cam_data.lens = focal_length_mm
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.sensor_width = sensor_width_mm
        cam_data.shift_x = 0.5 - cx / intrinsics["width"]
        cam_data.shift_y = 0.5 - cy / intrinsics["height"]
    else:
        print(f"Warning: Unsupported intrinsics model '{intrinsics['model']}', using defaults")

    cam_obj = bpy.data.objects.new(name, cam_data)
    cam_obj.location = location

    # Link to scene/collection
    if collection is not None:
        collection.objects.link(cam_obj)
    else:
        bpy.context.scene.collection.objects.link(cam_obj)

    # Add a Track To constraint so camera looks at target
    con = cam_obj.constraints.new(type="TRACK_TO")
    con.target = target_obj
    con.track_axis = "TRACK_NEGATIVE_Z"  # typical camera forward axis
    con.up_axis = "UP_Y"

    # Store custom properties on the camera *object* (not cam_data),
    # so export code can read obj["model"] etc.
    cam_data = cam_obj.data
    cam_data["model"] = intrinsics["model"]
    cam_data["width"] = int(intrinsics["width"])
    cam_data["height"] = int(intrinsics["height"])
    cam_data["params"] = list(map(float, intrinsics["params"]))

    return cam_obj


def spawn_cameras(c: SpawnCamerasConfig):
    # Validate target
    target = bpy.data.objects.get(c.LOOKUP_TARGET)
    if target is None:
        raise ValueError(f"LOOKUP_TARGET object '{c.LOOKUP_TARGET}' not found in bpy.data.objects")
    # Optional output collection
    out_col = _ensure_collection(c.CAMERA_COLLECTION_NAME) if c.USE_COLLECTION else None
    cam_global_index = 1

    if len(c.BEZIER_CURVE_LIST) != 0:
        bezier_curve_list = c.BEZIER_CURVE_LIST
    else:  # Fallback to all curves in the scene if none specified
        bezier_curve_list = [obj.name for obj in bpy.data.objects if obj.type == "CURVE"]
        print(f"No BEZIER_CURVE_LIST specified, using all curves in scene: {bezier_curve_list}")

    for curve_name in bezier_curve_list:
        curve_obj = bpy.data.objects.get(curve_name)
        if curve_obj is None:
            raise ValueError(f"Curve object '{curve_name}' not found in bpy.data.objects")
        if curve_obj.type != "CURVE":
            raise TypeError(f"'{curve_name}' is not a CURVE object (got type={curve_obj.type})")

        sampled = _sample_curve_world_points(curve_obj, c.SAMPLES_PER_BEZIER_SEGMENT)
        cum, total_len = _arc_length_parameterization(sampled)

        # Even spacing along arc length:
        # If NUMBER_OF_CAMERAS == 1 -> put it at start.
        if c.NUMBER_OF_CAMERAS <= 0:
            continue
        if c.NUMBER_OF_CAMERAS == 1:
            distances = [0.0]
        else:
            step = total_len / c.NUMBER_OF_CAMERAS
            distances = [k * step for k in range(c.NUMBER_OF_CAMERAS)]

        for j, d in enumerate(distances):
            p = _point_at_distance(sampled, cum, d)

            intr = get_initial_intrinsics(cam_global_index)
            # TODO@mateosss: reformat logic to always expect cam_000Number and
            # use000Number as global idx base, anything out of that is a warning
            # and not used
            # cam_name = f"cam{cam_global_index:03d}_{curve_obj.name}_{j:03d}"
            # _create_camera(cam_name, cam_global_index, p, target, intr, collection=out_col)
            cam_name = f"cam_{cam_global_index:04d}"
            _create_camera(cam_name, p, target, intr, collection=out_col)

            cam_global_index += 1


def clear_cameras(c: SpawnCamerasConfig):
    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA" and obj.name != c.ANIMATION_CAMERA]
    for cam in cameras:
        bpy.data.objects.remove(cam, do_unlink=True)


def apply_lookat():
    cameras = [obj for obj in bpy.data.objects if obj.type == "CAMERA"]
    for cam in cameras:
        bpy.context.view_layer.objects.active = cam
        bpy.ops.constraint.apply(constraint="Track To", owner="OBJECT")


def iter_action_fcurves(action: bpy.types.Action, anim_data: bpy.types.AnimData):
    """
    Yield every FCurve in *action* using the Blender 4.4+ layered-action API.

    In Blender 4.4+, FCurves are stored inside:
        action.layers  →  layer.strips  →  strip.channelbag(slot)  →  .fcurves

    We also try action_slot from anim_data so we get exactly the FCurves
    bound to this object. If that slot isn't found we fall back to iterating
    all channelbags on every strip.
    """
    slot = anim_data.action_slot if anim_data else None

    for layer in action.layers:
        for strip in layer.strips:
            # ── preferred: channelbag for the object's own slot ──────────────
            if slot is not None:
                try:
                    cb = strip.channelbag(slot)
                    if cb is not None:
                        yield from cb.fcurves
                        continue  # move to next strip
                except Exception:
                    pass  # slot not in this strip → fall through

            # ── fallback: walk every channelbag on the strip ─────────────────
            try:
                for cb in strip.channelbags:
                    yield from cb.fcurves
            except AttributeError:
                pass  # strip has no channelbags at all


def get_location_rotation_keyframes(obj: bpy.types.Object) -> list:
    """
    Return a sorted, deduplicated list of frame numbers that carry at least one
    keyframe on a location or rotation channel of *obj*.
    """
    anim_data = obj.animation_data
    action = anim_data.action if (anim_data and anim_data.action) else None
    if action is None:
        return []

    RELEVANT = {"location", "rotation_euler", "rotation_quaternion", "scale"}
    frames: set = set()

    for fcurve in iter_action_fcurves(action, anim_data):
        if fcurve.data_path in RELEVANT:
            for kp in fcurve.keyframe_points:
                frames.add(int(round(kp.co.x)))

    return sorted(frames)


def matrix_at_frame(obj: bpy.types.Object, frame: int) -> Matrix:
    """
    Return the world-space matrix of *obj* evaluated at *frame*.
    Evaluating the dependency graph honours constraints and drivers.
    """
    bpy.context.scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return obj.evaluated_get(depsgraph).matrix_world.copy()


def ensure_collection(name: str) -> bpy.types.Collection:
    """Return (or create) a top-level scene collection with *name*."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def spawn_static_camera(
    source: bpy.types.Object,
    frame: int,
    world_mat: Matrix,
    collection: bpy.types.Collection,
    index: int,
) -> bpy.types.Object:
    """
    Create a new camera object at *world_mat*, cloned from *source*,
    with no animation data, linked into *collection*.
    """
    cam_data = source.data.copy()
    cam_data.name = f"cam_{frame:04d}_DATA"

    new_obj = bpy.data.objects.new(
        name=f"cam_{frame:04d}",
        object_data=cam_data,
    )
    new_obj.matrix_world = world_mat
    new_obj.animation_data_clear()

    collection.objects.link(new_obj)
    return new_obj


def create_bezier_path(
    positions: list,
    name: str,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """
    Build a 3-D Bezier curve through *positions* with AUTO handles,
    linked into *collection*.
    """
    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 12
    curve_data.use_path = True

    spline = curve_data.splines.new(type="BEZIER")
    spline.bezier_points.add(len(positions) - 1)  # first point already exists

    for i, pos in enumerate(positions):
        bp = spline.bezier_points[i]
        bp.co = pos
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"

    curve_obj = bpy.data.objects.new(name=name, object_data=curve_data)
    collection.objects.link(curve_obj)
    return curve_obj


def spawn_animation_cameras(c: SpawnCamerasConfig):
    scene = bpy.context.scene

    # 1 ── Validate source camera ─────────────────────────────────────────────
    source_cam = bpy.data.objects.get(c.ANIMATION_CAMERA)
    if source_cam is None:
        raise ValueError(
            f"Object '{c.ANIMATION_CAMERA}' not found. " "Update c.ANIMATION_CAMERA at the top of this script."
        )
    if source_cam.type != "CAMERA":
        raise TypeError(f"'{c.ANIMATION_CAMERA}' is type '{source_cam.type}', expected 'CAMERA'.")
    if not (source_cam.animation_data and source_cam.animation_data.action):
        raise RuntimeError(f"Camera '{c.ANIMATION_CAMERA}' has no animation action attached.")

    # 2 ── Collect keyframe frames ─────────────────────────────────────────────
    frames = get_location_rotation_keyframes(source_cam)
    if not frames:
        raise RuntimeError(
            f"No location/rotation keyframes found on '{c.ANIMATION_CAMERA}'.\n"
            "Make sure the keyframes are on the object (not a constraint target)."
        )
    print(f"[spawner] {len(frames)} keyframe(s) found: {frames}")

    # 3 ── Prepare output collection ──────────────────────────────────────────
    col_name = f"CameraKeyframes_{c.ANIMATION_CAMERA}"
    col = ensure_collection(col_name)

    original_frame = scene.frame_current

    # 4 ── Spawn static cameras & collect positions ───────────────────────────
    positions = []
    for _, frame in enumerate(frames):
        mat = matrix_at_frame(source_cam, frame)
        positions.append(mat.translation.copy())
    for idx, frame in enumerate(frames[:: c.ANIMATION_STEP], start=1):
        mat = matrix_at_frame(source_cam, frame)
        cam_obj = spawn_static_camera(source_cam, idx, mat, col, idx)
        positions.append(mat.translation.copy())
        print(f"  frame {frame:5d}  →  {cam_obj.name}  loc={mat.translation}")

    scene.frame_set(original_frame)

    # 5 ── Create Bezier path ─────────────────────────────────────────────────
    path_name = f"{c.ANIMATION_CAMERA}_Path"
    path_obj = create_bezier_path(positions, path_name, col)

    print(
        f"\n[spawner] Done.\n"
        f"  {len(frames)} camera copies  →  collection '{col_name}'\n"
        f"  Bezier path '{path_name}'  ({len(positions)} control points)"
    )


def main():
    config = SpawnCamerasConfig()
    spawn_cameras(config)


if __name__ == "__main__":
    main()
