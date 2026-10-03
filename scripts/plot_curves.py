"""Read a run's CSV logs and write loss/accuracy plots under figures/."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _save(fig, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {dest}")


def plot_run(run_dir, out_dir):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    name = run_dir.name

    step_csv = run_dir / "step_log.csv"
    epoch_csv = run_dir / "epoch_log.csv"
    if not step_csv.exists() and not epoch_csv.exists():
        raise FileNotFoundError(f"No step_log.csv or epoch_log.csv in {run_dir}")

    if step_csv.exists():
        steps = pd.read_csv(step_csv)
        if not steps.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(steps["step"], steps["loss"], lw=1.0)
            ax.set_xlabel("step")
            ax.set_ylabel("train loss")
            ax.set_title(f"{name} — train loss")
            ax.grid(True, alpha=0.3)
            _save(fig, out_dir / f"{name}_loss.png")

    if epoch_csv.exists():
        epochs = pd.read_csv(epoch_csv)
        if epochs.empty:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs["epoch"], epochs["train_loss"], marker="o", label="train")
        if "val_loss" in epochs.columns and epochs["val_loss"].notna().any():
            ax.plot(epochs["epoch"], epochs["val_loss"], marker="o", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"{name} — epoch loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, out_dir / f"{name}_epoch_loss.png")

        if "val_acc" in epochs.columns and epochs["val_acc"].notna().any():
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(epochs["epoch"], epochs["val_acc"], marker="o", color="C2")
            ax.set_xlabel("epoch")
            ax.set_ylabel("val acc")
            ax.set_title(f"{name} — val accuracy")
            ax.grid(True, alpha=0.3)
            _save(fig, out_dir / f"{name}_acc.png")


def main():
    parser = argparse.ArgumentParser(description="Plot train/val curves from a run directory.")
    parser.add_argument("--run_dir", required=True, help="Path to a run folder with CSV logs")
    parser.add_argument("--out_dir", default=str(ROOT / "figures"))
    args = parser.parse_args()
    plot_run(args.run_dir, args.out_dir)


if __name__ == "__main__":
    main()
