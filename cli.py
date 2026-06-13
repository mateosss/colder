#!/usr/bin/env python3

import os

from pathlib import Path
from typing import Annotated
from typer import Argument, Exit, Typer, echo

from common import sh, shret

CONFIG_DEFAULT = "configs/empty.json"
REQUIREMENTS_FILE = "requirements.txt"

REPO_ROOT = Path(__file__).resolve().parent

# A lot of piping from uv to python to blender's python, so let's try keeping colors
os.environ["FORCE_COLOR"] = "1"


def blender_python_executable(blender_executable: str) -> str:
    out, ret = sh(f'{blender_executable} --background --python-expr "import sys;print(sys.executable)"')
    if ret != 0:
        echo("Failed to determine Blender's Python executable.")
        raise Exit(code=ret)

    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        echo("Blender did not report a Python executable.")
        raise Exit(code=1)

    return lines[0]


app = Typer()


@app.command()
def generate(
    blend_file: Annotated[Path, Argument(help="Path to the .blend file to open.")],
    config: Annotated[Path, Argument(help="Path to a JSON file with ExportSceneConfig fields.")] = CONFIG_DEFAULT,
    blender: Annotated[str, Argument(help="Blender executable to run.")] = "blender",
) -> None:
    if not blend_file.exists():
        echo(f"Blend file not found: {blend_file}")
        raise Exit(code=1)

    if not config.exists():
        echo(f"Config file not found: {config}")
        raise Exit(code=1)

    blender_entrypoint = REPO_ROOT / "blender" / "cli_entry.py"
    retcode = shret(f"{blender} --background {blend_file} --python {blender_entrypoint} -- --config {config}")

    raise Exit(code=retcode)


@app.command(name="setup_blender")
def setup_blender(blender: Annotated[str, Argument(help="Blender executable to use.")] = "blender") -> None:
    requirements = REPO_ROOT / REQUIREMENTS_FILE

    freeze_out, freeze_ret = sh("uv pip freeze")
    if freeze_ret != 0:
        raise Exit(code=freeze_ret)

    requirements.write_text(freeze_out, encoding="utf-8")

    blender_python = blender_python_executable(blender)
    install_ret = shret(f'"{blender_python}" -m pip install -r {requirements}')

    print(f"{requirements=}")
    print(f"{blender_python=}")
    print("Done.")
    raise Exit(code=install_ret)


if __name__ == "__main__":
    app()
