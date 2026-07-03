from __future__ import annotations

import pickle
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any


class CrossPlatformPathUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "pathlib":
            if name == "PosixPath":
                return PurePosixPath
            if name == "WindowsPath":
                return PureWindowsPath
        return super().find_class(module, name)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return CrossPlatformPathUnpickler(f).load()


def json_ready(value: Any) -> Any:
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return tuple(json_ready(v) for v in value)
    return value
