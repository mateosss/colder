# Blender 5.0.1
# Export a synthetic COLMAP sparse model by projecting mesh vertices into cameras.
#
# Expects cameras to have custom properties:
#   model (str), width (int), height (int), params (list of floats)
#
# Output:
#   EXPORT_PATH/sparse/0/{rigs.txt,cameras.txt,frames.txt,images.txt,points3D.txt}
#
# COLMAP format reference: https://colmap.github.io/format.html

import os
import random
from math import radians
from mathutils import Vector, Matrix, Quaternion
from dataclasses import dataclass, field

import bpy


@dataclass
class ExportSceneConfig:
    EXPORT_PATH: str = "colmap_export"
    TARGET_OBJECTS: list[str] = field(default_factory=list)  # empty=list means all mesh objects

    POINT_3D_SAVE_NOISE_STDEV: float = 0.0  # in meters
    POINT_2D_SAVE_NOISE_STDEV: float = 0.0  # in pixels
    POSE_TRANSLATION_NOISE_STDEV: float = 0.0  # in meters
    POSE_ROTATION_NOISE_STDEV: float = 0.0  # in degrees

    POINT_3D_DENSITY: float = 1.0  # fraction of vertices to keep, 1 for all
    POINT_2D_DENSITY: float = 1.0  # fraction of observations to keep, 1 for all
    MIN_NUM_OBS_PER_POINT3D: int = 2

    # Image name pattern in images.txt (COLMAP typically wants actual image filenames; this is synthetic)
    IMAGE_NAME_FMT: str = "cam_{:04d}.png"


# Fix random seed
random.seed(42)


# -------------------------
# Projection models
# -------------------------
def project_pinhole(x, y, z, width, height, fx, fy, cx, cy):
    # Camera coords: x right, y down, z forward (COLMAP convention)
    if z <= 0.0:
        return (False, 0.0, 0.0)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    valid = (0.0 <= u < float(width)) and (0.0 <= v < float(height))
    return (valid, u, v)


def project_simple_radial(x, y, z, width, height, f, cx, cy, k1):
    # Camera coords: x right, y down, z forward (COLMAP convention)
    if z <= 0.0:
        return (False, 0.0, 0.0)
    x_n = x / z
    y_n = y / z
    r2 = x_n * x_n + y_n * y_n
    radial_distortion = 1.0 + k1 * r2
    x_d = x_n * radial_distortion
    y_d = y_n * radial_distortion
    u = f * x_d + cx
    v = f * y_d + cy
    valid = (0.0 <= u < float(width)) and (0.0 <= v < float(height))
    return (valid, u, v)


CAMERA_PROJECT = {
    "PINHOLE": project_pinhole,
    "SIMPLE_RADIAL": project_simple_radial,
}


# -------------------------
# Data structures
# -------------------------
class CameraRec:
    def __init__(self, camera_id, obj):
        self.camera_id = camera_id  # also used as RIG_ID and FRAME_ID
        self.obj = obj  # bpy.types.Object (type CAMERA)
        self.model = obj.data["model"]
        self.width = obj.data["width"]
        self.height = obj.data["height"]
        self.params = obj.data["params"]

    def project(self, x, y, z):
        assert self.model in CAMERA_PROJECT, f"Unsupported camera model '{self.model}'"
        proj_func = CAMERA_PROJECT[self.model]
        return proj_func(x, y, z, self.width, self.height, *self.params)


class Point3DRec:
    def __init__(self, point3d_id, xyz, rgb):
        self.point3d_id = point3d_id
        self.xyz = xyz  # Vector world
        self.rgb = rgb  # (r,g,b) uint8
        self.error = 0.0  # unknown; leave 0
        self.track = []  # list of (image_id, point2d_idx)


class ImageRec:
    def __init__(self, c: ExportSceneConfig, image_id, camera_id, name, qvec, tvec):
        self.image_id = image_id
        self.camera_id = camera_id
        self.name = name
        self.qvec = self._noisy_rotation(c, Quaternion(qvec))  # (qw,qx,qy,qz) world->cam
        self.tvec = self._noisy_translation(c, Vector(tvec))  # (tx,ty,tz) world->cam
        self.points2d = []  # list of (x, y, point3d_id)

    def _noisy_translation(self, c: ExportSceneConfig, vec: Vector) -> tuple:
        if c.POSE_TRANSLATION_NOISE_STDEV == 0.0:
            return (vec.x, vec.y, vec.z)
        dx = random.gauss(0.0, c.POSE_TRANSLATION_NOISE_STDEV)
        dy = random.gauss(0.0, c.POSE_TRANSLATION_NOISE_STDEV)
        dz = random.gauss(0.0, c.POSE_TRANSLATION_NOISE_STDEV)
        return (vec.x + dx, vec.y + dy, vec.z + dz)

    def _noisy_rotation(self, c: ExportSceneConfig, quat: Quaternion) -> tuple:
        if c.POSE_ROTATION_NOISE_STDEV == 0.0:
            return (quat.w, quat.x, quat.y, quat.z)
        axis = Vector((random.gauss(0.0, 1.0), random.gauss(0.0, 1.0), random.gauss(0.0, 1.0)))
        axis.normalize()
        angle = random.gauss(0.0, radians(c.POSE_ROTATION_NOISE_STDEV))
        dq = Quaternion(axis, angle)  # axis-angle to quaternion
        q_orig = quat
        q_noisy = dq @ q_orig
        if q_noisy.w < 0.0:
            q_noisy.w, q_noisy.x, q_noisy.y, q_noisy.z = -q_noisy.w, -q_noisy.x, -q_noisy.y, -q_noisy.z
        return (q_noisy.w, q_noisy.x, q_noisy.y, q_noisy.z)


class ColmapProblem:
    def __init__(self, config: ExportSceneConfig):
        self.config = config
        self.cameras = []  # list[CameraRec]
        self.images = []  # list[ImageRec]
        self.points3d = []  # list[Point3DRec]

    def save(self, export_path: str):
        model_dir = os.path.join(export_path, "sparse", "0")
        os.makedirs(model_dir, exist_ok=True)

        self._write_rigs(os.path.join(model_dir, "rigs.txt"))
        self._write_cameras(os.path.join(model_dir, "cameras.txt"))
        self._write_frames(os.path.join(model_dir, "frames.txt"))
        self._write_images(os.path.join(model_dir, "images.txt"))
        self._write_points3d(os.path.join(model_dir, "points3D.txt"))

    def _write_rigs(self, path):
        # Trivial rig per camera, no sensors[] pose extras.
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Rig calib list with one line of data per calib:\n")
            f.write(
                "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])\n"
            )
            f.write(f"# Number of rigs: {len(self.cameras)}\n")
            for cam in self.cameras:
                # one sensor: CAMERA <id>
                f.write(f"{cam.camera_id} 1 CAMERA {cam.camera_id}\n")

    def _write_cameras(self, path):
        # Camera list per COLMAP.
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(self.cameras)}\n")
            for cam in self.cameras:
                params_str = " ".join(str(p) for p in cam.params)
                f.write(f"{cam.camera_id} {cam.model} {cam.width} {cam.height} {params_str}\n")

    def _write_frames(self, path):
        # Frame uses RIG_FROM_WORLD pose; with 1 sensor per rig, DATA_IDS is "1 CAMERA <cam_id> <image_id>".
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Frame list with one line of data per frame:\n")
            f.write(
                "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n"
            )
            f.write(f"# Number of frames: {len(self.images)}\n")
            for img in self.images:
                qw, qx, qy, qz = img.qvec
                tx, ty, tz = img.tvec
                f.write(
                    f"{img.image_id} {img.camera_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 CAMERA {img.camera_id} {img.image_id}\n"
                )

    def _write_images(self, path):
        # Two lines per image: header + points2D triplets.
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(self.images)}\n")
            for img in self.images:
                qw, qx, qy, qz = img.qvec
                tx, ty, tz = img.tvec
                f.write(f"{img.image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {img.camera_id} {img.name}\n")
                if img.points2d:
                    rows = []
                    for x, y, pid in img.points2d:
                        nx, ny = self._noisy_xy(x, y)
                        rows.append(f"{nx} {ny} {pid}")
                    f.write(" ".join(rows) + "\n")
                else:
                    f.write("\n")

    def _write_points3d(self, path):
        # One line per point with TRACK[] = (IMAGE_ID, POINT2D_IDX).
        with open(path, "w", encoding="utf-8") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {len(self.points3d)}\n")
            for p in self.points3d:
                r, g, b = p.rgb
                track_str = " ".join(f"{iid} {kidx}" for (iid, kidx) in p.track)
                nx, ny, nz = self._noisy_xyz(p.xyz)
                f.write(f"{p.point3d_id} {nx} {ny} {nz} {r} {g} {b} {p.error} {track_str}\n")

    def _noisy_xyz(self, vec: Vector):
        if self.config.POINT_3D_SAVE_NOISE_STDEV == 0.0:
            return (vec.x, vec.y, vec.z)
        dx = random.gauss(0.0, self.config.POINT_3D_SAVE_NOISE_STDEV)
        dy = random.gauss(0.0, self.config.POINT_3D_SAVE_NOISE_STDEV)
        dz = random.gauss(0.0, self.config.POINT_3D_SAVE_NOISE_STDEV)
        return (vec.x + dx, vec.y + dy, vec.z + dz)

    def _noisy_xy(self, x: float, y: float):
        if self.config.POINT_2D_SAVE_NOISE_STDEV == 0.0:
            return (x, y)
        dx = random.gauss(0.0, self.config.POINT_2D_SAVE_NOISE_STDEV)
        dy = random.gauss(0.0, self.config.POINT_2D_SAVE_NOISE_STDEV)
        return (x + dx, y + dy)


# -------------------------
# Helpers
# -------------------------

# Blender world to colmap world change of basis
# NOTE: in reality this looks like blender cam to colmap cam
T_B_C = Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),  # X -> X
        (0.0, -1.0, 0.0, 0.0),  # Y -> -Y
        (0.0, 0.0, -1.0, 0.0),  # Z -> -Z
        (0.0, 0.0, 0.0, 1.0),
    )
)
T_C_B = T_B_C.inverted()


def blender_to_colmap_pose(obj: bpy.types.Object):
    """
    Get Object (O) transform/pose in COLMAP (C) world coords from Blender (B) world coords.

    Sandwich proof:
    We have a transform T:=T_B_O that does the following for points in B frame
    [1] p'_B = T @ p_B
    We want an equivalent T* that does the same in C frame
    [2] p'_C = T* @ p_C
    Considering we can use T_B_C to express p'_B and p_B in C frame in [1]:
    [3] (T_B_C @ p'_C) = T @ (T_B_C @ p_C)
    [4] p'_C = (T_C_B @ T @ T_B_C) @ p_C // Therefore from [2]:
    [5] T* = T_C_B @ T @ T_B_C // This is the equivalent transform!

    """
    T = obj.matrix_world  # T_B_O
    return T_C_B @ T @ T_B_C


def mat_to_qt(W2C: Matrix):
    R = W2C.to_3x3()
    t = W2C.to_translation()
    q = R.to_quaternion()  # (w,x,y,z) Hamilton, matches COLMAP expectation.
    if q.w < 0.0:
        q.w, q.x, q.y, q.z = -q.w, -q.x, -q.y, -q.z
    return (q.w, q.x, q.y, q.z), (t.x, t.y, t.z)


def get_mesh_vertex_world_positions_and_colors(c: ExportSceneConfig, obj: bpy.types.Object):
    if obj.type != "MESH":
        raise TypeError(f"{obj.name} is not a MESH object")

    mesh = obj.data
    T_B_O = obj.matrix_world

    vtx_rgb = None

    # Find a color attribute of type FLOAT_COLOR or BYTE_COLOR
    col_attr = None
    if hasattr(mesh, "color_attributes"):
        for attr in mesh.color_attributes:
            if attr.data_type in {"FLOAT_COLOR", "BYTE_COLOR"}:
                col_attr = attr
                break

    if c.POINT_3D_DENSITY < 1.0:
        idxs = random.sample(range(len(mesh.vertices)), int(len(mesh.vertices) * c.POINT_3D_DENSITY))
    else:
        idxs = range(len(mesh.vertices))

    if col_attr is not None:
        if col_attr.domain == "POINT":  # glb/ply vertex colors
            # one color per vertex
            vtx_rgb = [tuple(col_attr.data[i].color) for i in idxs]
        elif col_attr.domain == "CORNER":  # vertex paint
            # average loop colors per vertex
            accum = [Vector((0.0, 0.0, 0.0)) for _ in mesh.vertices]
            cnt = [0 for _ in mesh.vertices]
            for poly in mesh.polygons:
                for li in poly.loop_indices:
                    vi = mesh.loops[li].vertex_index
                    c = col_attr.data[li].color
                    accum[vi] += Vector((c[0], c[1], c[2]))
                    cnt[vi] += 1
            vtx_rgb = []
            for i in range(len(mesh.vertices)):
                if cnt[i] > 0:
                    c = accum[i] / float(cnt[i])
                    vtx_rgb.append((c.x, c.y, c.z, 1.0))
                else:
                    vtx_rgb.append((0.0, 0.0, 0.0, 1.0))
        else:
            raise ValueError(f"Unsupported color attribute domain '{col_attr.domain}'")
    else:
        print(f"Warning: mesh '{obj.name}' has no color attribute; vertex colors will be black")

    verts = []
    for i, idx in enumerate(idxs):
        v = mesh.vertices[idx]
        pw = T_C_B @ T_B_O @ v.co  # in colmap world
        if vtx_rgb is not None:
            c = vtx_rgb[i]
            r = int(max(0, min(255, round(c[0] * 255.0))))
            g = int(max(0, min(255, round(c[1] * 255.0))))
            b = int(max(0, min(255, round(c[2] * 255.0))))
        else:
            r = g = b = 0
        verts.append((idx, pw, (r, g, b)))
    return verts


# -------------------------
# Main build
# -------------------------
def build_problem(c: ExportSceneConfig):
    prob = ColmapProblem(c)

    # Gather cameras (all camera objects in scene, sorted by name for determinism)
    cam_objs = [o for o in bpy.context.scene.objects if o.type == "CAMERA"]
    cam_objs.sort(key=lambda o: o.name)

    # COLMAP ids are typically 1-based in text examples (not required, but common).
    for idx, co in enumerate(cam_objs, start=1):
        prob.cameras.append(CameraRec(camera_id=idx, obj=co))

    # Create images = one per camera (for now we dont support rigs/frames)
    for cam in prob.cameras:
        T_C_O = T_C_B @ cam.obj.matrix_world @ T_B_C
        T_O_C = T_C_O.inverted()
        qvec, tvec = mat_to_qt(T_O_C)
        img_id = cam.camera_id
        name = c.IMAGE_NAME_FMT.format(img_id)
        prob.images.append(ImageRec(c, img_id, cam.camera_id, name, qvec, tvec))

    # Gather points from meshes
    points = []
    base_id = 1

    # Fallback to all curves in the scene if none specified
    if len(c.TARGET_OBJECTS) == 0:
        target_objects = [obj.name for obj in bpy.data.objects if obj.type == "MESH"]
        print(f"No TARGET_OBJECTS specified, using all meshes in scene: {target_objects}")
    else:
        target_objects = c.TARGET_OBJECTS

    for obj_name in target_objects:
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise ValueError(f"{target_objects=} contains '{obj_name}' but it was not found")
        verts = get_mesh_vertex_world_positions_and_colors(c, obj)
        for _, pw, rgb in verts:
            # Create a unique POINT3D_ID. Using sequential IDs avoids collisions across multiple objects.
            points.append((base_id, pw, rgb))
            base_id += 1

    # Initialize Point3D records
    prob.points3d = [Point3DRec(pid, pw, rgb) for (pid, pw, rgb) in points]

    # Observations: for each image, build points2d list; also populate point tracks.
    # POINT2D_IDX is the zero-based index into that image’s points2d list.
    obs_count = [0] * (len(prob.points3d) + 1)  # obs count per point3d_id
    for img in prob.images:
        cam = prob.cameras[img.camera_id - 1]
        T_C_O = T_C_B @ cam.obj.matrix_world @ T_B_C
        T_O_C = T_C_O.inverted()

        if c.POINT_2D_DENSITY < 1.0:
            prob_points = random.sample(prob.points3d, int(len(prob.points3d) * c.POINT_2D_DENSITY))
        else:
            prob_points = prob.points3d
        for p in prob_points:
            p_O = T_O_C @ p.xyz.to_4d()
            x, y, z = p_O.x, p_O.y, p_O.z
            ok, u, v = cam.project(x, y, z)

            if not ok:
                continue
            point2d_idx = len(img.points2d)  # zero-based
            img.points2d.append((u, v, p.point3d_id))
            p.track.append((img.image_id, point2d_idx))
            obs_count[p.point3d_id] += 1

    # Filter out points with less than 2 observations
    valid_point_ids = set()
    for p in prob.points3d:
        if obs_count[p.point3d_id] >= c.MIN_NUM_OBS_PER_POINT3D:
            valid_point_ids.add(p.point3d_id)
    prob.points3d = [p for p in prob.points3d if p.point3d_id in valid_point_ids]

    # Also filter image points2d to only keep those pointing to valid points
    for img in prob.images:
        img.points2d = [pt for pt in img.points2d if pt[2] in valid_point_ids]

    return prob


def export_scene(config: ExportSceneConfig):
    prob = build_problem(config)
    prob.save(config.EXPORT_PATH)


def main():
    config = ExportSceneConfig()
    export_scene(config)


if __name__ == "__main__":
    main()
