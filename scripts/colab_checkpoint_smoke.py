"""Drive checkpoint smoke test: write, read-back, corrupt latest, resume earlier."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.config import load_config  # noqa: E402


def _run(cmd: list[str], *, label: str) -> None:
    print(f"{label}: {' '.join(cmd)}", flush=True)
    try:
        subprocess.check_call(cmd, cwd=ROOT)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"{label} failed with exit code {error.returncode}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/colab_checkpoint_smoke.yaml",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/content/cs604-course-data"))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random tensors (recommended on Colab; avoids train.tar staging).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the smoke output directory before phase 1.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output = Path(cfg["output_dir"])
    if args.fresh and output.exists():
        print("removing previous smoke output", output, flush=True)
        shutil.rmtree(output)

    train = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/train.py"),
        "--config",
        str(args.config),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(output),
        "--max-steps",
        str(int(cfg.get("max_steps", 4))),
    ]
    use_synthetic = args.synthetic or bool(cfg.get("synthetic"))
    if not use_synthetic:
        train_txt = args.data_root / "train.txt"
        train_dir = args.data_root / "train"
        if not train_txt.is_file() or not train_dir.is_dir():
            print(
                f"Missing staged data under {args.data_root} (need train.txt and train/). "
                "Re-run stage_data or pass --synthetic.",
                flush=True,
            )
            raise SystemExit(1)
    else:
        train.append("--synthetic")

    _run(train, label="phase 1 initial training")

    checkpoints = output / "checkpoints"
    numbered = sorted(checkpoints.glob("resume_step_*.pt"))
    if len(numbered) < 2:
        raise SystemExit(
            f"Expected at least two resume checkpoints in {checkpoints}; "
            f"found {[p.name for p in numbered]}. Check {output}/run.json and metrics.csv."
        )
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

    resume_cmd = train + ["--resume", str(previous)]
    _run(resume_cmd, label="phase 2 resume")
    last = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    print(f"resume complete last.pt step={last['step']}", flush=True)
    print("colab checkpoint smoke passed", flush=True)


if __name__ == "__main__":
    main()
