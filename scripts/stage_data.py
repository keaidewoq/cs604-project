"""Put the CS604 archives on a fast local disk and extract them.

Google Drive is the cache. Training reads images from ``--local-dir``
(Colab's ``/content`` disk, or the desktop dataset folder).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "doem1997/cs604-course-data"
ALL_ARCHIVES = ("train.tar.gz", "val.tar.gz")

TRAIN_TXT_MIN_LINES = 350_000
VAL_TXT_MIN_LINES = 40_000

ARCHIVE_CHECKS: dict[str, tuple[tuple[str, ...], int | None]] = {
    "train.tar.gz": (("train.txt", "train"), TRAIN_TXT_MIN_LINES),
    "val.tar.gz": (("val.txt", "val"), VAL_TXT_MIN_LINES),
}


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def extract_marker(local_dir: Path, archive_name: str) -> Path:
    return local_dir / f".extracted_{archive_name}"


def archive_layout_ok(local_dir: Path, archive_name: str) -> bool:
    spec = ARCHIVE_CHECKS.get(archive_name)
    if spec is None:
        return True
    required, min_lines = spec
    for name in required:
        if not (local_dir / name).exists():
            return False
    list_file = next((local_dir / name for name in required if name.endswith(".txt")), None)
    if list_file is not None and min_lines is not None:
        if count_lines(list_file) < min_lines:
            return False
    return True


def clear_extract_marker(local_dir: Path, archive_name: str) -> None:
    extract_marker(local_dir, archive_name).unlink(missing_ok=True)


def report_disk(path: Path) -> None:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / (1024**3)
    total_gib = usage.total / (1024**3)
    print(f"disk {path}: {free_gib:.1f} GiB free of {total_gib:.1f} GiB total", flush=True)


def resolve_archive_path(local_dir: Path, downloaded: Path, name: str) -> Path:
    candidate = local_dir / name
    if candidate.is_file():
        return candidate
    if downloaded.is_file():
        return downloaded
    raise FileNotFoundError(f"Could not locate {name} after download (got {downloaded})")


def _ensure_archive(name: str, local_dir: Path, drive_dir: Path | None) -> Path:
    local = local_dir / name
    if local.is_file():
        return local
    if drive_dir is not None and (drive_dir / name).is_file():
        print(f"copying {name} from Drive to local disk", flush=True)
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_dir / name, local)
        return local

    from huggingface_hub import hf_hub_download

    print(f"downloading {name} from {REPO_ID} (Drive cache miss)", flush=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=name,
            local_dir=str(local_dir),
        )
    )
    archive = resolve_archive_path(local_dir, downloaded, name)
    if drive_dir is not None:
        drive_dir.mkdir(parents=True, exist_ok=True)
        cached = drive_dir / name
        if not cached.is_file():
            print(f"caching {name} on Drive for the next session", flush=True)
            shutil.copy2(archive, cached)
    return archive


def _extract(archive: Path, local_dir: Path) -> None:
    marker = extract_marker(local_dir, archive.name)
    if marker.is_file() and archive_layout_ok(local_dir, archive.name):
        print(f"already extracted: {archive.name}", flush=True)
        return
    if marker.is_file():
        print(
            f"stale or incomplete extract for {archive.name}; removing marker and re-extracting",
            flush=True,
        )
        marker.unlink(missing_ok=True)
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing archive {archive}. Upload to Drive cs604/data/ or allow Hugging Face download."
        )
    print(f"extracting {archive.name}", flush=True)
    try:
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(local_dir)], check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"tar failed for {archive.name} (exit {error.returncode})") from error
    if not archive_layout_ok(local_dir, archive.name):
        marker.unlink(missing_ok=True)
        list_path = local_dir / "train.txt" if archive.name == "train.tar.gz" else local_dir / "val.txt"
        lines = count_lines(list_path) if list_path.is_file() else 0
        raise SystemExit(
            f"After extracting {archive.name}, layout check failed under {local_dir}. "
            f"train.txt/val.txt lines seen: {lines}. Archive may be corrupt or disk full."
        )
    marker.write_text("ok\n", encoding="utf-8")


def validate_training_layout(local_dir: Path) -> None:
    train_txt = local_dir / "train.txt"
    if not train_txt.is_file():
        raise SystemExit(
            f"Missing {train_txt}. Run stage_data with a valid train.tar.gz "
            "(Drive cs604/data/ or Hugging Face download)."
        )
    lines = count_lines(train_txt)
    if lines < TRAIN_TXT_MIN_LINES:
        raise SystemExit(
            f"{train_txt} has only {lines} lines (expected at least {TRAIN_TXT_MIN_LINES}). "
            "Delete stale .extracted_* markers and re-run stage_data."
        )
    if not (local_dir / "train").is_dir():
        raise SystemExit(f"Missing image tree {local_dir / 'train'} after extract.")
    print(f"train split ok: {lines} lines in train.txt", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--drive-dir", type=Path)
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Stage only train.tar.gz (enough for LoRA training; saves VM disk).",
    )
    args = parser.parse_args(argv)

    args.local_dir.mkdir(parents=True, exist_ok=True)
    report_disk(args.local_dir)

    archives = ("train.tar.gz",) if args.train_only else ALL_ARCHIVES
    for name in archives:
        archive = _ensure_archive(name, args.local_dir, args.drive_dir)
        _extract(archive, args.local_dir)
        if (
            args.drive_dir is not None
            and archive.is_file()
            and (args.drive_dir / name).is_file()
        ):
            archive.unlink()
            print(f"removed local copy of {name}; Drive still has the archive", flush=True)

    validate_training_layout(args.local_dir)
    print("data ready", args.local_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
