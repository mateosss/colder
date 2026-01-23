# Blender 5.0.1
# Spawns cameras along one or more Bezier curve objects, evenly spaced by arc length,
# and makes each camera look at a target object.
#
# Usage:
# - Put this in Blender's Text Editor and Run Script.
# - Ensure the curve objects and target object exist in the scene.

import bpy
from mathutils import Vector
from mathutils.geometry import interpolate_bezier  # available in Blender's mathutils

# -------------------------
# Script options (edit me)
# -------------------------
BEZIER_CURVE_LIST = []  # names of curve objects to use, leave empty to use all curves in scene
LOOKUP_TARGET = "Empty"  # object name all cameras will look at
NUMBER_OF_CAMERAS = 10

# Sampling density for arc-length approximation (higher = more accurate, slower)
SAMPLES_PER_BEZIER_SEGMENT = 64

# If True, creates/uses a collection to keep things tidy
USE_COLLECTION = True
CAMERA_COLLECTION_NAME = "SpawnedCameras"


def get_initial_intrinsics(_: int) -> dict:
    # parameter is intentionally unused for now; keep it for future per-camera variation.
    return {
        "model": "SIMPLE_RADIAL",
        "width": 640,
        "height": 480,
        "params": [420, 640 / 2, 480 / 2, 0.0],  # f, cx, cy, k1
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


def main():
    # Validate target
    target = bpy.data.objects.get(LOOKUP_TARGET)
    if target is None:
        raise ValueError(f"LOOKUP_TARGET object '{LOOKUP_TARGET}' not found in bpy.data.objects")

    # Optional output collection
    out_col = _ensure_collection(CAMERA_COLLECTION_NAME) if USE_COLLECTION else None
    cam_global_index = 0

    if len(BEZIER_CURVE_LIST) != 0:
        bezier_curve_list = BEZIER_CURVE_LIST
    else: # Fallback to all curves in the scene if none specified
        bezier_curve_list = [obj.name for obj in bpy.data.objects if obj.type == "CURVE"]
        print(f"No BEZIER_CURVE_LIST specified, using all curves in scene: {bezier_curve_list}")

    for curve_name in bezier_curve_list:
        curve_obj = bpy.data.objects.get(curve_name)
        if curve_obj is None:
            raise ValueError(f"Curve object '{curve_name}' not found in bpy.data.objects")
        if curve_obj.type != "CURVE":
            raise TypeError(f"'{curve_name}' is not a CURVE object (got type={curve_obj.type})")

        sampled = _sample_curve_world_points(curve_obj, SAMPLES_PER_BEZIER_SEGMENT)
        cum, total_len = _arc_length_parameterization(sampled)

        # Even spacing along arc length:
        # If NUMBER_OF_CAMERAS == 1 -> put it at start.
        if NUMBER_OF_CAMERAS <= 0:
            continue
        if NUMBER_OF_CAMERAS == 1:
            distances = [0.0]
        else:
            step = total_len / NUMBER_OF_CAMERAS
            distances = [k * step for k in range(NUMBER_OF_CAMERAS)]

        for j, d in enumerate(distances):
            p = _point_at_distance(sampled, cum, d)

            intr = get_initial_intrinsics(cam_global_index)
            cam_name = f"Cam_{curve_obj.name}_{j:03d}"
            _create_camera(cam_name, p, target, intr, collection=out_col)

            cam_global_index += 1


if __name__ == "__main__":
    main()
