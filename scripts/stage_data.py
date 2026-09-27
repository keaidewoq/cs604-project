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
ARCHIVES = ("train.tar.gz", "val.tar.gz")


def _ensure_archive(name: str, local_dir: Path, drive_dir: Path | None) -> Path:
    local = local_dir / name
    if local.is_file():
        return local
    if drive_dir is not None and (drive_dir / name).is_file():
        print(f"copying {name} from Drive to local disk")
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_dir / name, local)
        return local

    from huggingface_hub import hf_hub_download

    print(f"downloading {name} from {REPO_ID}")
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=name,
            local_dir=str(local_dir),
        )
    )
    if drive_dir is not None:
        drive_dir.mkdir(parents=True, exist_ok=True)
        cached = drive_dir / name
        if not cached.is_file():
            print(f"caching {name} on Drive for the next session")
            shutil.copy2(downloaded, cached)
    return downloaded


def _extract(archive: Path, local_dir: Path) -> None:
    marker = local_dir / f".extracted_{archive.name}"
    if marker.is_file():
        print(f"already extracted: {archive.name}")
        return
    print(f"extracting {archive.name}")
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(local_dir)], check=True)
    marker.write_text("ok\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--drive-dir", type=Path)
    args = parser.parse_args()

    for name in ARCHIVES:
        archive = _ensure_archive(name, args.local_dir, args.drive_dir)
        _extract(archive, args.local_dir)
        if args.drive_dir is not None and archive.is_file() and (args.drive_dir / name).is_file():
            # The extracted tree is what training reads. Drop the local tar to save VM disk.
            archive.unlink()
            print(f"removed local copy of {name}; Drive still has the archive")
    print("data ready", args.local_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
