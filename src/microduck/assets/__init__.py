from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent


def resolve_model_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ASSETS_DIR / path

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MJCF 文件不存在：{path}")

    return path