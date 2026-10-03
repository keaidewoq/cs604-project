"""Download the course dataset from Hugging Face and extract archives."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "doem1997/cs604-course-data"


def extract_archives(root):
    """Extract zip/tar archives sitting directly under `root`."""
    root = Path(root)
    archives = []
    for path in sorted(root.iterdir()):
        name = path.name.lower()
        if path.is_file() and (
            name.endswith(".zip")
            or name.endswith(".tar")
            or name.endswith(".tar.gz")
            or name.endswith(".tgz")
        ):
            archives.append(path)

    for path in archives:
        print(f"[download] extracting {path.name}")
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                zf.extractall(root)
        else:
            with tarfile.open(path) as tf:
                tf.extractall(root)


def hint_data_root(root):
    root = Path(root)
    if (root / "train.txt").exists() or (root / "train").is_dir():
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "train.txt").exists() or (child / "train").is_dir():
            print(f"[download] data looks nested at {child} — set data_root to this path")
            return child
    return root


def main():
    parser = argparse.ArgumentParser(description="Download CS604 course images from Hugging Face.")
    parser.add_argument("--dest", default="/content/data", help="Local directory (default: /content/data)")
    parser.add_argument("--repo_id", default=REPO_ID)
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] snapshot {args.repo_id} -> {dest}")
    snapshot_download(repo_id=args.repo_id, repo_type="dataset", local_dir=str(dest))
    extract_archives(dest)
    resolved = hint_data_root(dest)
    print(f"[download] ready at {resolved}")


if __name__ == "__main__":
    main()
