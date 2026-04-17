from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return expand_env(data)


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def iter_json_files(input_dir: str | Path, glob_pattern: str = "*.json") -> Iterable[Path]:
    yield from sorted(Path(input_dir).glob(glob_pattern))
