"""Train the ViT baseline. Local and Colab both call this script."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.engine.loop import run_training  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key in ("data_root", "output_dir", "max_steps", "limit_samples", "batch_size", "resume"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = str(value) if key in {"data_root", "output_dir", "resume"} else value
    if args.synthetic:
        cfg["synthetic"] = True
    run_training(cfg)


if __name__ == "__main__":
    main()
