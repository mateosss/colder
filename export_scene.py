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
import bpy
from mathutils import Vector, Matrix

# -------------------------
# Script options (edit me)
# -------------------------
EXPORT_PATH = "colmap_export"
RENDER_OBJECTS = ["Suzanne"]

# If True, only keep points that are observed in >= 2 images (often required for meaningful SfM)
FILTER_MIN_TRACK_LEN_2 = False

# Image name pattern in images.txt (COLMAP typically wants actual image filenames; this is synthetic)
IMAGE_NAME_FMT = "cam_{:04d}.png"


# -------------------------
# Projection models
# -------------------------
def project_pinhole(x, y, z, fx, fy, cx, cy, width, height):
    # Camera coords: x right, y down, z forward (COLMAP convention)
    if z <= 0.0:
        return (False, 0.0, 0.0)
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    valid = (0.0 <= u < float(width)) and (0.0 <= v < float(height))
    return (valid, u, v)


CAMERA_PROJECT = {
    "PINHOLE": project_pinhole,
    # "SIMPLE_RADIAL": project_simple_radial,
}


# -------------------------
# Data structures
# -------------------------
class CameraRec:
    def __init__(self, camera_id, obj):
        self.camera_id = camera_id          # also used as RIG_ID and FRAME_ID
        self.obj = obj                      # bpy.types.Object (type CAMERA)
        self.model = str(obj.get("model", "PINHOLE"))
        self.width = int(obj.get("width", 640))
        self.height = int(obj.get("height", 480))
        self.params = list(obj.get("params", [300.0, 300.0, 320.0, 240.0]))


class Point3DRec:
    def __init__(self, point3d_id, xyz, rgb):
        self.point3d_id = point3d_id
        self.xyz = xyz          # Vector world
        self.rgb = rgb          # (r,g,b) uint8
        self.error = 0.0        # unknown; leave 0
        self.track = []         # list of (image_id, point2d_idx)


class ImageRec:
    def __init__(self, image_id, camera_id, name, qvec, tvec):
        self.image_id = image_id
        self.camera_id = camera_id
        self.name = name
        self.qvec = qvec        # (qw,qx,qy,qz) world->cam
        self.tvec = tvec        # (tx,ty,tz) world->cam
        self.points2d = []      # list of (x, y, point3d_id)


class ColmapProblem:
    def __init__(self):
        self.cameras = []       # list[CameraRec]
        self.images = []        # list[ImageRec]
        self.points3d = []      # list[Point3DRec]

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
            f.write("#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])\n")
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
            f.write("#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n")
            f.write(f"# Number of frames: {len(self.images)}\n")
            for img in self.images:
                qw, qx, qy, qz = img.qvec
                tx, ty, tz = img.tvec
                f.write(f"{img.image_id} {img.camera_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 CAMERA {img.camera_id} {img.image_id}\n")

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
                    f.write(" ".join(f"{x} {y} {pid}" for (x, y, pid) in img.points2d) + "\n")
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
                f.write(f"{p.point3d_id} {p.xyz.x} {p.xyz.y} {p.xyz.z} {r} {g} {b} {p.error} {track_str}\n")


# -------------------------
# Helpers
# -------------------------
def blender_cam_world_to_colmap_cam(world_to_cam: Matrix):
    # Convert Blender 4x4 world->cam into COLMAP qvec/tvec (Hamilton quaternion, w,x,y,z)
    R = world_to_cam.to_3x3()
    t = world_to_cam.to_translation()

    q = R.to_quaternion()  # mathutils Quaternion: (w, x, y, z)
    # Ensure a consistent sign (COLMAP doesn't require it, but it avoids flips)
    if q.w < 0.0:
        q.w, q.x, q.y, q.z = -q.w, -q.x, -q.y, -q.z
    return (q.w, q.x, q.y, q.z), (t.x, t.y, t.z)


def get_world_to_camera_matrix(cam_obj: bpy.types.Object):
    # Blender gives camera-to-world at matrix_world; invert to get world-to-camera.
    return cam_obj.matrix_world.inverted()


def get_mesh_vertex_world_positions_and_colors(obj: bpy.types.Object):
    if obj.type != 'MESH':
        raise TypeError(f"{obj.name} is not a MESH object")

    mesh = obj.data
    mw = obj.matrix_world

    # Build per-vertex color if possible; Blender colors are typically per-loop.
    vtx_rgb = None
    if hasattr(mesh, "color_attributes") and mesh.color_attributes:
        col_attr = mesh.color_attributes.active or mesh.color_attributes[0]
        # Try to support common domains: 'POINT' or 'CORNER'
        if col_attr.domain == 'POINT':
            vtx_rgb = [col_attr.data[i].color[:] for i in range(len(mesh.vertices))]
        elif col_attr.domain == 'CORNER':
            # accumulate loop colors into vertices
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

    verts = []
    for i, v in enumerate(mesh.vertices):
        pw = mw @ v.co
        if vtx_rgb is not None:
            c = vtx_rgb[i]
            r = int(max(0, min(255, round(c[0] * 255.0))))
            g = int(max(0, min(255, round(c[1] * 255.0))))
            b = int(max(0, min(255, round(c[2] * 255.0))))
        else:
            r, g, b = 0, 0, 0
        verts.append((i, pw, (r, g, b)))
    return verts


def project_world_point(cam: CameraRec, world_to_cam: Matrix, Pw: Vector):
    Pc = world_to_cam @ Pw.to_4d()
    x, y, z = Pc.x, Pc.y, Pc.z

    proj = CAMERA_PROJECT.get(cam.model)
    if proj is None:
        raise ValueError(f"Unsupported camera model '{cam.model}' (no projector in CAMERA_PROJECT)")

    if cam.model == "PINHOLE":
        fx, fy, cx, cy = cam.params[:4]
        return proj(x, y, z, fx, fy, cx, cy, cam.width, cam.height)

    raise ValueError(f"Projector dispatch missing implementation for '{cam.model}'")


# -------------------------
# Main build
# -------------------------
def build_problem():
    prob = ColmapProblem()

    # Gather cameras (all camera objects in scene, sorted by name for determinism)
    cam_objs = [o for o in bpy.context.scene.objects if o.type == "CAMERA"]
    cam_objs.sort(key=lambda o: o.name)

    # COLMAP ids are typically 1-based in text examples (not required, but common).
    for idx, co in enumerate(cam_objs, start=1):
        prob.cameras.append(CameraRec(camera_id=idx, obj=co))

    # Create images = one per camera (“SfM not SLAM” single frame per camera convention)
    for cam in prob.cameras:
        W2C = get_world_to_camera_matrix(cam.obj)
        qvec, tvec = blender_cam_world_to_colmap_cam(W2C)
        img_id = cam.camera_id
        name = IMAGE_NAME_FMT.format(img_id)
        prob.images.append(ImageRec(img_id, cam.camera_id, name, qvec, tvec))

    # Gather points from meshes
    points = []
    base_id = 1
    for obj_name in RENDER_OBJECTS:
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            raise ValueError(f"RENDER_OBJECTS contains '{obj_name}' but it was not found")
        verts = get_mesh_vertex_world_positions_and_colors(obj)
        for (vidx, pw, rgb) in verts:
            # Create a unique POINT3D_ID. Using sequential IDs avoids collisions across multiple objects.
            points.append((base_id, pw, rgb))
            base_id += 1

    # Initialize Point3D records
    prob.points3d = [Point3DRec(pid, pw, rgb) for (pid, pw, rgb) in points]

    # Observations: for each image, build points2d list; also populate point tracks.
    # POINT2D_IDX is the zero-based index into that image’s points2d list.
    for img in prob.images:
        cam = prob.cameras[img.camera_id - 1]
        W2C = get_world_to_camera_matrix(cam.obj)

        for p in prob.points3d:
            ok, u, v = project_world_point(cam, W2C, p.xyz)
            if not ok:
                continue
            point2d_idx = len(img.points2d)  # zero-based
            img.points2d.append((u, v, p.point3d_id))
            p.track.append((img.image_id, point2d_idx))

    if FILTER_MIN_TRACK_LEN_2:
        prob.points3d = [p for p in prob.points3d if len(p.track) >= 2]
        # Note: this does not remove now-dangling observations from images.txt.
        # If this is enabled, add a cleanup pass (left out to keep script short).

    return prob


def main():
    prob = build_problem()
    prob.save(EXPORT_PATH)


if __name__ == "__main__":
    main()
