"""Load experiment YAML files and expand the machine-specific paths in them.

Configs write paths as ``${CS604_DATA}/...`` and ``${CS604_OUTPUTS}/...`` so
the same file works on every teammate's machine. Set the environment variables
to point elsewhere; unset, they default to the sibling folders
``../cs604-course-data`` and ``../cs604-outputs`` next to the repo.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH_DEFAULTS = {
    "CS604_DATA": (ROOT.parent / "cs604-course-data").as_posix(),
    "CS604_OUTPUTS": (ROOT.parent / "cs604-outputs").as_posix(),
}
_VARIABLE = re.compile(r"\$\{(\w+)\}")


def _lookup(match: re.Match[str]) -> str:
    name = match.group(1)
    value = os.environ.get(name) or PATH_DEFAULTS.get(name)
    if value is None:
        raise SystemExit(f"Config uses ${{{name}}}, but the environment variable is not set")
    return value.replace("\\", "/")


def expand_paths(value: Any) -> Any:
    """Replace ``${VAR}`` in every string of a loaded config."""
    if isinstance(value, str):
        return _VARIABLE.sub(_lookup, value)
    if isinstance(value, list):
        return [expand_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_paths(item) for key, item in value.items()}
    return value


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return expand_paths(cfg)
