import sys
import importlib
from pathlib import Path
import bpy

bl_info = {
    "name": "COLDER - Colmap Export Helper",
    "author": "Mateo de Mayo",
    "blender": (4, 0, 0),
    "description": "Tools to help export Blender scenes and spawn cameras in COLMAP format",
    "location": "View3D > Sidebar > COLDER",
    "category": "3D View",
}


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


class COLDER_OT_spawn_cameras(bpy.types.Operator):
    bl_idname = "colder.spawn_cameras"
    bl_label = "Spawn cameras"

    def execute(self, context):
        run_module_main("spawn_cameras")
        return {"FINISHED"}


class COLDER_OT_export_scene(bpy.types.Operator):
    bl_idname = "colder.export_scene"
    bl_label = "Export scene"

    def execute(self, context):
        run_module_main("export_scene")
        return {"FINISHED"}


class COLDER_OT_clear_cameras(bpy.types.Operator):
    bl_idname = "colder.clear_cameras"
    bl_label = "Clear cameras"

    def execute(self, context):
        run_module_main("clear_cameras")
        return {"FINISHED"}


class COLDER_PT_panel(bpy.types.Panel):
    bl_idname = "COLDER_PT_panel"
    bl_label = "COLDER Tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "COLDER"  # N-panel tab name

    def draw(self, context):
        layout = self.layout
        layout.operator("colder.spawn_cameras", text="Spawn cameras")
        layout.operator("colder.export_scene", text="Export scene")
        layout.operator("colder.clear_cameras", text="Clear cameras")


classes = (
    COLDER_OT_spawn_cameras,
    COLDER_OT_export_scene,
    COLDER_OT_clear_cameras,
    COLDER_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


def main():
    register()
