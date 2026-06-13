import sys
from pathlib import Path
from typing import Annotated
from typer import Typer, Option
from export_scene import ExportSceneConfig, generate_all


def ensure_repo_root_on_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


app = Typer()


@app.command()
def run(config: Annotated[Path, Option("--config", help="Path to the JSON export config.")]) -> None:
    config = ExportSceneConfig.from_json_file(config)
    generate_all(config)
    print(f"COLDER export finished for: {config.EXPORT_PATH}")


def main(argv: list[str] | None = None) -> int:
    ensure_repo_root_on_syspath()

    if argv is None:
        # running under Blender: strip args before "--" so Typer only sees script args
        if "--" in sys.argv:
            idx = sys.argv.index("--")
            sys.argv = [sys.argv[0]] + sys.argv[idx + 1 :]
        app()
        return 0

    # programmatic invocation: extract script args after "--" and call Typer app
    script_argv = argv[argv.index("--") + 1 :] if "--" in argv else argv
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + script_argv
        app()
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
