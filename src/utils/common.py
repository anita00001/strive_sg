from pathlib import Path
from types import SimpleNamespace
from typing import Any


def dict_to_namespace(data: dict[str, Any]) -> SimpleNamespace:
    """
    Convert a dictionary into an object that supports attribute access.

    Example:
        cfg = dict_to_namespace({"batch_size": 4})
        print(cfg.batch_size)
    """
    return SimpleNamespace(**data)


def mkdir(path: str | Path) -> Path:
    """
    Create a directory, including any missing parent directories.

    Calling this function on an existing directory is safe.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path