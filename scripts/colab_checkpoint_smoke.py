"""Drive checkpoint smoke test: write, read-back, corrupt latest, resume earlier."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/colab_checkpoint_smoke.yaml",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/content/cs604-course-data"))
    args = parser.parse_args()

    train = [
        sys.executable,
        str(ROOT / "scripts/train.py"),
        "--config",
        str(args.config),
        "--data-root",
        str(args.data_root),
    ]
    print("phase 1: initial training", " ".join(train), flush=True)
    subprocess.check_call(train)

    import yaml

    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    output = Path(cfg["output_dir"])
    checkpoints = output / "checkpoints"
    numbered = sorted(checkpoints.glob("resume_step_*.pt"))
    if len(numbered) < 2:
        raise SystemExit(f"Expected at least two resume checkpoints in {checkpoints}")
    previous, newest = numbered[-2], numbered[-1]
    for path in (newest, output / "last.pt"):
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        print(f"verified {path.name} step={loaded['step']}", flush=True)

    with newest.open("r+b") as handle:
        handle.truncate(max(1, handle.tell() // 2))
    print(f"truncated {newest.name}", flush=True)
    try:
        torch.load(newest, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"truncated file fails as expected: {type(error).__name__}", flush=True)
    else:
        raise SystemExit("Truncated checkpoint still loaded; smoke test failed")

    resume_cmd = train + ["--resume", str(previous), "--output-dir", str(output)]
    print("phase 2: resume from", previous.name, flush=True)
    subprocess.check_call(resume_cmd)
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    print(f"resume complete last.pt step={last['step']}", flush=True)
    print("colab checkpoint smoke passed", flush=True)


if __name__ == "__main__":
    main()
