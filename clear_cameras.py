# Clear all cameras in the scene
import bpy

def clear_cameras():
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    for cam in cameras:
        bpy.data.objects.remove(cam, do_unlink=True)

def main():
    clear_cameras()
