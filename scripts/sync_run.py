"""Copy finished Colab runs from Google Drive into cs604-outputs and verify them.

Google Drive for desktop shows My Drive as a local folder, so the Colab output
folder ``My Drive/cs604/outputs/<run>`` is readable here (default ``H:/My Drive``;
override with --drive-root or the CS604_DRIVE_OUTPUTS environment variable).

Every file except the numbered ``checkpoints/*_step_*.pt`` copies is copied and
checked by SHA-256. ``last.pt``/``best.pt``/``baseline.pt`` must also match the
newest hash that training logged in ``checkpoint_hashes.jsonl``, which catches a
half-synced Drive file. ``--prune-drive`` then deletes the numbered checkpoint
copies on Drive (they go to the Drive trash; empty it to free quota).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.engine.checkpoint_io import sha256_file  # noqa: E402

DEFAULT_DRIVE = Path(os.environ.get("CS604_DRIVE_OUTPUTS", "H:/My Drive/cs604/outputs"))
DEFAULT_LOCAL = ROOT.parent / "cs604-outputs"
FINISHED = ("completed", "evaluated")
SKIP_DIRS = ("checkpoints",)


def logged_hashes(run_dir: Path) -> dict[str, str]:
    """Newest logged SHA-256 per checkpoint file name (last.pt, best.pt, ...)."""

    manifest = run_dir / "checkpoint_hashes.jsonl"
    hashes: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                hashes[Path(entry["dest"]).name] = entry["sha256"]
    return hashes


def files_to_copy(source: Path) -> list[Path]:
    return sorted(
        path.relative_to(source)
        for path in source.rglob("*")
        if path.is_file() and path.relative_to(source).parts[0] not in SKIP_DIRS
    )


def sync_run(source: Path, dest: Path, *, allow_incomplete: bool = False) -> dict[str, str]:
    """Copy one run folder and return {relative path: sha256} of what was verified."""

    run_json = source / "run.json"
    if not run_json.is_file():
        raise SystemExit(f"{source} has no run.json; is the name right and Drive synced?")
    status = json.loads(run_json.read_text(encoding="utf-8")).get("status", "running")
    if status not in FINISHED and not allow_incomplete:
        raise SystemExit(f"{source.name}: status is {status!r}, not finished. Wait, or pass --allow-incomplete.")

    logged = logged_hashes(source)
    verified: dict[str, str] = {}
    for rel in files_to_copy(source):
        src_file, dst_file = source / rel, dest / rel
        digest = sha256_file(src_file)
        if rel.name in logged and rel.parent == Path(".") and logged[rel.name] != digest:
            raise SystemExit(
                f"{src_file} does not match the hash training logged; Drive may still be syncing. Retry later."
            )
        if dst_file.is_file():
            if sha256_file(dst_file) != digest:
                raise SystemExit(f"{dst_file} already exists with different content; refusing to overwrite.")
        else:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            partial = dst_file.with_name(dst_file.name + ".partial")
            shutil.copy2(src_file, partial)
            if sha256_file(partial) != digest:
                partial.unlink(missing_ok=True)
                raise SystemExit(f"Copy of {src_file} is corrupt; retry.")
            partial.replace(dst_file)
        verified[rel.as_posix()] = digest

    missing = [name for name in logged if not (dest / name).is_file()]
    if missing:
        raise SystemExit(f"{source.name}: logged checkpoints missing on Drive: {missing}")
    record = {
        "source": str(source),
        "synced_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "files": verified,
    }
    (dest / "sync.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return verified


def prune_numbered_checkpoints(source: Path) -> int:
    """Delete Drive's numbered checkpoint copies; returns bytes removed."""

    removed = 0
    for path in sorted((source / "checkpoints").glob("*_step_*.pt")):
        removed += path.stat().st_size
        path.unlink()
    return removed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run folder names under --drive-root (= EXPERIMENT)")
    parser.add_argument("--drive-root", type=Path, default=DEFAULT_DRIVE)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--prune-drive", action="store_true",
                        help="after verifying, delete numbered checkpoints/*_step_*.pt on Drive")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="copy a run whose status is not completed (never prunes it)")
    args = parser.parse_args(argv)

    if not args.drive_root.is_dir():
        raise SystemExit(
            f"{args.drive_root} not found. Install Google Drive for desktop, or pass --drive-root "
            "(or set CS604_DRIVE_OUTPUTS) to where cs604/outputs is visible."
        )
    for name in args.runs:
        source, dest = args.drive_root / name, args.local_root / name
        verified = sync_run(source, dest, allow_incomplete=args.allow_incomplete)
        print(f"{name}: {len(verified)} files verified in {dest}", flush=True)
        if args.prune_drive:
            if args.allow_incomplete:
                print(f"{name}: not pruning an incomplete run", flush=True)
                continue
            removed = prune_numbered_checkpoints(source)
            print(f"{name}: removed {removed / 1024**3:.2f} GiB of numbered checkpoints from Drive "
                  "(empty the Drive trash to free quota)", flush=True)


if __name__ == "__main__":
    main()
