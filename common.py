import os
import sys
from contextlib import contextmanager
import bpy
import importlib
from pathlib import Path

DEPTHS_DIR = "depths"
IMAGES_DIR = "images"


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


@contextmanager
def stdout_redirected(to=os.devnull):
    """Redirect stdout to the given file or file-like object.
    Thanks: https://blender.stackexchange.com/questions/44560/how-to-supress-bpy-render-messages-in-terminal-output
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w", encoding="utf-8") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
