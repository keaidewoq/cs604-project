"""Copy a run's small record files into the repo's outputs/ folder for git.

A run folder in ``cs604-outputs/<run>`` (or ``$CS604_OUTPUTS``) holds two kinds
of files. The small records (``run.json``, ``config.json``, ``metrics.csv``,
``baseline.json``, logs, ``*.inference.json``, ...) are copied to
``outputs/<run>/`` here and committed. The large files (checkpoints, split
lists, prediction files) stay in the run folder and are shared through the team
Google Drive folder; ``outputs/<run>/artifacts.json`` lists them with their
sizes so teammates know what to fetch.

Absolute paths inside the copied text are rewritten to ``${CS604_OUTPUTS}``,
``${CS604_DATA}``, or ``~`` so no machine-specific path is committed.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.config import PATH_DEFAULTS  # noqa: E402

DEFAULT_LOCAL = Path(os.environ.get("CS604_OUTPUTS", PATH_DEFAULTS["CS604_OUTPUTS"]))
DEFAULT_REPO_OUTPUTS = ROOT / "outputs"
MAX_BYTES = 1024 * 1024
# Small records worth reviewing in git. Keep in sync with outputs/ in .gitignore.
RECORD_PATTERNS = (
    "run.json",
    "config.json",
    "metrics.csv",
    "baseline.json",
    "hardware.json",
    "checkpoint_audit.json",
    "checkpoint_hashes.jsonl",
    "sync.json",
    "leaderboard.json",
    "class_prototypes.json",
    "*.log",
    "analysis/*.json",
    "submissions/*.inference.json",
)


def is_record(relative: Path) -> bool:
    return any(fnmatch.fnmatch(relative.as_posix(), pattern) for pattern in RECORD_PATTERNS)


def path_spellings(path: Path) -> list[str]:
    """The ways one folder can appear in text: forward slashes, backslashes, JSON-escaped."""

    spellings = []
    # As given and resolved: on Windows they differ for 8.3 short names (VANLE~1).
    for posix in dict.fromkeys((Path(path).as_posix(), Path(path).resolve().as_posix())):
        windows = posix.replace("/", "\\")
        spellings += [posix, windows.replace("\\", "\\\\"), windows]
    return spellings


def generalise(text: str, local_root: Path) -> str:
    data_root = Path(os.environ.get("CS604_DATA", PATH_DEFAULTS["CS604_DATA"]))
    for folder, token in ((local_root, "${CS604_OUTPUTS}"), (data_root, "${CS604_DATA}"), (Path.home(), "~")):
        for spelling in path_spellings(folder):
            text = text.replace(spelling, token)
    return text


def publish_run(source: Path, dest: Path, local_root: Path) -> tuple[list[str], list[dict]]:
    if not (source / "run.json").is_file() and not (source / "class_prototypes.json").is_file():
        raise SystemExit(f"{source} is not a run folder (no run.json or class_prototypes.json)")
    copied, left_behind = [], []
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        relative = path.relative_to(source)
        size = path.stat().st_size
        if is_record(relative) and size <= MAX_BYTES:
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            text = path.read_text(encoding="utf-8", errors="replace")
            target.write_text(generalise(text, local_root), encoding="utf-8", newline="")
            copied.append(relative.as_posix())
        elif not relative.parts[0] == "checkpoints" or relative.name in ("last.pt", "best.pt"):
            left_behind.append({"path": relative.as_posix(), "bytes": size})
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run": source.name,
        "location": f"team Drive folder CS604_RUNS/{source.name}/ (and the owner's cs604-outputs/{source.name}/)",
        "note": "Large files are not in git. Numbered checkpoints/*_step_*.pt copies are omitted.",
        "files": left_behind,
    }
    (dest / "artifacts.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return copied, left_behind


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run folder names under --local-root (= EXPERIMENT)")
    parser.add_argument("--all", action="store_true", help="publish every run folder under --local-root")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--repo-outputs", type=Path, default=DEFAULT_REPO_OUTPUTS)
    args = parser.parse_args(argv)

    names = list(args.runs)
    if args.all:
        names += sorted(
            p.name for p in args.local_root.iterdir()
            if (p / "run.json").is_file() or (p / "class_prototypes.json").is_file()
        )
    if not names:
        parser.error("name at least one run, or pass --all")
    for name in dict.fromkeys(names):
        copied, left = publish_run(args.local_root / name, args.repo_outputs / name, args.local_root)
        big = sum(item["bytes"] for item in left) / 1024**3
        print(f"{name}: {len(copied)} record files -> outputs/{name}/; "
              f"{len(left)} large files ({big:.2f} GiB) stay on Drive", flush=True)


if __name__ == "__main__":
    main()
