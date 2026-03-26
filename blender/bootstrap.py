# NOTE: This is a copy of the script living inside of the blend file to load COLDER

# Refresh COLDER code

import sys
import importlib
from pathlib import Path
import bpy


def main():
    script_dir = Path(bpy.data.filepath).parent
    if script_dir not in sys.path:
        sys.path.append(str(script_dir))

    # scripts = ["main", "export_scene", "spawn_cameras"]
    scripts = [] # .py scripts in the same directory as this .blend file
    for file in script_dir.iterdir():
        if file.suffix == ".py":
            scripts.append(file.stem)

    # Reload all scripts and collect the main module (which registers COLDER operators)
    main_mod = None
    for script in scripts:
        mod = importlib.import_module(script)
        importlib.reload(mod) # refresh if edited
        if script == "main":
            main_mod = mod

    # Run main module to register COLDER
    assert main_mod is not None, "main.py not found in the same directory as the .blend file"
    main_mod.main()


if __name__ == "__main__":
    main()
