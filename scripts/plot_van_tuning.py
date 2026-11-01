"""Slide figures for Van's full-FT tuning rounds (Van holdout, laptop, one change at a time).

Numbers from report/van_full_ft_findings.md (branch van-448-finals), sections 4.2 to 4.4:
  lr_sweep      Round A: top backbone lr, 1 epoch, LP-FT start, layer decay 0.75
  head_decay    Round B: head lr 10x and no layer decay, per epoch over a 3-epoch run
  augmentation  Round C: one change each from head lr 10x (71.37), 3 epochs

    python scripts/plot_van_tuning.py --out_dir figures
writes van_<name>.png and .svg. Same look as scripts/plot_full_ft_ablation.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from plot_full_ft_ablation import INK, INK_2, NEUTRAL, PAGE, RECIPE, REMOVED, RULE

GRID = "#e8e6df"
TIE = 0.4  # seed spread 0.33 pp: gaps under ~0.4 are ties


def style(ax):
    ax.set_facecolor(PAGE)
    ax.tick_params(length=0, pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)


def save(fig, out, note):
    fig.text(0.5, 0.03, note, ha="center", va="bottom", fontsize=11, color=INK_2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
        print(f"[plot] wrote {out.with_suffix('.' + ext)}")
    plt.close(fig)


def lr_sweep(out):
    lrs = [5e-5, 1e-4, 2e-4, 4e-4, 8e-4]
    acc = [66.24, 66.82, 67.22, 67.47, 67.34]
    chosen = 3
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=PAGE)
    style(ax)
    ax.set_xscale("log")
    ax.plot(lrs, acc, color=NEUTRAL[1], lw=2, zorder=2)
    for i, (x, y) in enumerate(zip(lrs, acc)):
        fill, edge = RECIPE if i == chosen else NEUTRAL
        ax.scatter(x, y, s=150 if i == chosen else 90, facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=3)
        ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 13), textcoords="offset points", ha="center",
                    color=INK, fontweight="bold" if i == chosen else "normal")
    ax.annotate("chosen: 4e-4\n(8e-4 ties but was\nunstable early)", (lrs[chosen], acc[chosen]), xytext=(-20, -150),
                textcoords="offset points", ha="left", color=INK_2, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.8))
    ax.set_xticks(lrs, ["5e-5", "1e-4", "2e-4", "4e-4", "8e-4"])
    ax.minorticks_off()
    ax.set_xlim(3.8e-5, 1.05e-3)
    ax.set_ylim(65.8, 68.0)
    ax.set_xlabel("Learning rate of the top block (lower blocks scaled by layer decay 0.75)", color=INK_2)
    ax.set_ylabel("Holdout top-1 (%)", color=INK_2)
    save(fig, out, "Van holdout, 1 epoch, LP-FT start · the team's earlier full FT used 1e-5, 40× lower")


def head_decay(out):
    epochs = [0, 1, 2, 3]
    start = 60.37  # linear-probe head before any fine-tuning
    # (label, accuracy per epoch, colour, label nudge so the end labels don't collide)
    runs = [("Control (head shares backbone lr)", [start, 63.86, 68.57, 69.83], NEUTRAL[1], -0.9),
            ("+ head lr 10× backbone", [start, 62.18, 69.20, 71.37], RECIPE[1], 0.9),
            ("− remove layer decay", [start, 44.90, 58.44, 63.12], REMOVED[1], 0)]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=PAGE)
    style(ax)
    ax.axhline(start, xmax=0.56, color=RULE, lw=0.8, ls=(0, (4, 4)), zorder=1)
    ax.text(3.12, start, f"{start:.1f}  starting point (linear probe)", va="center", ha="left", color=INK_2)
    for label, acc, colour, nudge in runs:
        ax.plot(epochs, acc, color=colour, lw=2, zorder=2)
        ax.scatter(epochs[1:], acc[1:], s=70, facecolor=PAGE, edgecolor=colour, linewidth=1.5, zorder=3)
        ax.text(3.12, acc[-1] + nudge, f"{acc[-1]:.1f}  {label}", va="center", ha="left", color=INK)
    ax.annotate("drops below the\nstarting point", (1, 44.90), xytext=(22, -4), textcoords="offset points",
                va="center", color=INK_2, fontsize=11)
    ax.set_xticks(epochs, ["start", "1", "2", "3"])
    ax.set_xlim(-0.1, 5.4)
    ax.set_ylim(42, 74)
    ax.spines["bottom"].set_bounds(0, 3)
    ax.set_xlabel("Epoch (of a 3-epoch run)", color=INK_2)
    ax.set_ylabel("Holdout top-1 (%)", color=INK_2)
    save(fig, out, "Van holdout, LP-FT start, top lr 4e-4 · compare final epochs: mid-run scores are low "
         "while the lr is still high")


def augmentation(out):
    ref = 71.37
    rows = [("Reference: head lr 10×", ref, NEUTRAL),
            ("Crop scale [0.35, 1] (not [0.08, 1])", 71.11, NEUTRAL),
            ("Colour jitter off", 71.29, NEUTRAL),
            ("Head lr 30× (not 10×)", 71.97, NEUTRAL)]
    fig, ax = plt.subplots(figsize=(9, 4.0), facecolor=PAGE)
    style(ax)
    ax.grid(False)
    ys = list(range(len(rows)))[::-1]
    ax.axvspan(ref - TIE, ref + TIE, ymax=0.86, color=GRID, zorder=0, lw=0)
    ax.text(ref, ys[0] + 0.55, f"within noise of the reference (±{TIE})", ha="center", va="bottom",
            color=INK_2, fontsize=11)
    for y, (label, acc, (fill, edge)) in zip(ys, rows):
        ax.scatter(acc, y, s=110, facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=3)
        delta = "" if acc == ref else f" ({acc - ref:+.1f})".replace("-", "−")
        ax.annotate(f"{acc:.1f}{delta}", (acc, y), xytext=(12, 0), textcoords="offset points", va="center",
                    color=INK)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlim(70.4, 72.6)
    ax.set_xticks([70.5, 71.0, 71.5, 72.0, 72.5])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Holdout top-1 (%)", color=INK_2)
    save(fig, out, "Van holdout, 3 epochs, LP-FT start · head lr 30× ties 10× once the head starts "
         "from prototypes (72.7 vs 73.0)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13,
                         "text.color": INK, "xtick.color": INK_2, "ytick.color": INK})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lr_sweep(out / "van_lr_sweep")
    head_decay(out / "van_head_layer_decay")
    augmentation(out / "van_augmentation")


if __name__ == "__main__":
    main()
