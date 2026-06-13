#!/usr/bin/env python3

from pathlib import Path
from common import shout
from typing import Annotated
from typer import Typer, Option, Argument, echo, Exit

CONFIG_DEFAULT = "empty.json"


def build_command(blender_executable: str, blend_file: Path, config_file: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parent
    blender_entrypoint = repo_root / "blender" / "cli_entry.py"
    return [
        blender_executable,
        "-b",
        str(blend_file),
        "-P",
        str(blender_entrypoint),
        "--",
        "--config",
        str(config_file),
    ]


app = Typer()


@app.command()
def main(
    blend_file: Annotated[Path, Argument(help="Path to the .blend file to open.")],
    config: Annotated[Path, Argument(help="Path to a JSON file with ExportSceneConfig fields.")] = CONFIG_DEFAULT,
    blender: Annotated[str, Option(help="Blender executable to run.")] = "blender",
) -> None:

    if not blend_file.exists():
        echo(f"Blend file not found: {blend_file}")
        raise Exit(code=1)

    if not config.exists():
        echo(f"Config file not found: {config}")
        raise Exit(code=1)

    blender_entrypoint = Path(__file__).resolve().parent / "blender" / "cli_entry.py"
    command = f"{blender} --background {blend_file} --python {blender_entrypoint} -- --config {config}"
    ret = shout(command)
    raise Exit(code=ret)


if __name__ == "__main__":
    app()
