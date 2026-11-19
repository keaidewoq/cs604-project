"""Slide figures: higher training resolution beat more epochs at equal compute (Task 1, full FT).

Three separate experiments:
  1. Anastasia, ~2,000-class subset, 29 min per arm, augreg2: 224 vs 224->336 vs 336
     (progress_log.md, "Resolution ablation", ~2026-10-02).
  2. Van, equal-compute screen (Van holdout, 10 training images/class, equal GB-h per arm,
     crop 0.9): full FT 336/384/448/512 and AdaptFormer 336/384/448
     (final_report_draft.md / presentation_storyboard.md slide 7).
  3. CodaBench val, full FT with the tuned recipe, all data, 144 min: 224/336/384/448
     (presentation_storyboard.md slide 7).

    python scripts/plot_resolution.py --out_dir figures
writes {png,svg} of resolution_subset (1), resolution_equal_compute (2), resolution_plateau
(2, full FT only), resolution_codabench (3) and resolution_combined (1-3 on one axis).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plot_full_ft_ablation import INK, INK_2, NEUTRAL, PAGE, RECIPE, RULE

ADAPT = ("#e6f1fb", "#378add")  # second series (AdaptFormer), same pale-fill / dark-edge pairing

# (label, top-1, epochs, style); augreg2, best-epoch holdout top-1
SUBSET = [
    ("224 px", 81.03, 19, NEUTRAL),
    ("224 → 336 px (last 15%)", 82.87, 18, NEUTRAL),
    ("336 px", 84.72, 9, RECIPE),
]
# training size -> (holdout top-1, epochs that fit the budget)
FULL_FT = {336: (63.20, 7.7), 384: (64.06, 5.5), 448: (64.79, 4.1), 512: (64.98, 2.9)}
ADAPTFORMER = {336: (59.63, None), 384: (61.18, None), 448: (62.72, None)}
# CodaBench plain, full FT with Van's tuned recipe, all data, 144 min A100: 224 Van
# (full_ft_224px_proto-cosine_llrd0.75_top-lr4e-4_2.4h), 336 Anastasia (full_ft_336_llrd), 384 / 448 Van (fused AdamW + compile)
CODABENCH = {224: 75.7, 336: 80.7, 384: 81.7, 448: 82.5}
SUBSET_LINE = {224: 81.03, 336: 84.72}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, pad=8)


def subset_figure(out: Path):
    axis_start, axis_end = 78, 87
    bar_h = 0.62
    fig, ax = plt.subplots(figsize=(10, 3.6), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    ax.set_xlim(axis_start, axis_end)
    ys = [-i for i in range(len(SUBSET))]
    ax.set_ylim(ys[-1] - 0.6, 0.6)

    fig.canvas.draw()
    bbox = ax.get_window_extent()
    px_per_x = bbox.width / (axis_end - axis_start)
    px_per_y = bbox.height / (1.2 + len(SUBSET) - 1)
    for y, (label, acc, epochs, (fill, edge)) in zip(ys, SUBSET):
        ax.add_patch(FancyBboxPatch((axis_start, y - bar_h / 2), acc - axis_start, bar_h,
                                    boxstyle=f"round,pad=0,rounding_size={7 / px_per_x}",
                                    mutation_aspect=px_per_x / px_per_y,
                                    facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=2))
        delta = "" if label == "224 px" else f" ({acc - SUBSET[0][1]:+.1f})"
        ax.text(acc + 0.12, y, f"{acc:.1f}{delta}", va="center", ha="left", color=INK)
        ax.text(axis_start + 0.15, y, f"{epochs} epochs", va="center", ha="left", color=INK_2,
                fontsize=11, zorder=3)

    ax.set_yticks(ys, [r[0] for r in SUBSET])
    ax.set_xticks(range(axis_start, axis_end + 1, 2))
    style(ax)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Holdout top-1 (%)", color=INK_2, labelpad=8)
    fig.text(0.01, 0.02, "Full FT, augreg2, ~2,000-class subset, 5 holdout photos/class, 29 min per arm "
             "(A100). Run-to-run noise ~0.1 pp.", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
    plt.close(fig)


def equal_compute_figure(out: Path):
    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    for series, name, (fill, edge) in ((FULL_FT, "Full FT", RECIPE), (ADAPTFORMER, "AdaptFormer", ADAPT)):
        xs = list(series)
        accs = [series[x][0] for x in xs]
        ax.plot(xs, accs, color=edge, lw=2, zorder=2)
        ax.scatter(xs, accs, s=80, color=fill, edgecolors=edge, linewidths=1.5, zorder=3)
        for x, (acc, epochs) in series.items():
            text = f"{acc:.1f}" if epochs is None else f"{acc:.1f}\n{epochs} ep"
            ax.annotate(text, (x, acc), xytext=(0, 12), textcoords="offset points",
                        ha="center", va="bottom", color=INK, fontsize=11, linespacing=1.1)
        ax.annotate(name, (xs[-1], accs[-1]), xytext=(14, -4), textcoords="offset points",
                    ha="left", va="center", color=edge, fontsize=13, fontweight="bold")

    ax.text(480, 64.6, "+0.2: flat", ha="center", va="top", color=INK_2, fontsize=11)
    ax.annotate("(512 not run)", (448, 62.72), xytext=(14, -22), textcoords="offset points",
                ha="left", va="center", color=INK_2, fontsize=11)

    ax.set_xticks([336, 384, 448, 512], ["336", "384", "448", "512"])
    ax.set_xlim(310, 560)
    ax.set_ylim(58.5, 67)
    ax.set_xlabel("Training resolution (px)", color=INK_2, labelpad=8)
    ax.set_ylabel("Holdout top-1 (%)", color=INK_2, labelpad=8)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    style(ax)
    fig.text(0.01, 0.02, "Van's equal-compute screen: same GPU budget per arm, so higher resolution "
             "fits fewer epochs. 10 training photos/class, crop 0.9.", color=INK_2, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
    plt.close(fig)


def combined_figure(out: Path):
    """All three full-FT experiments on one resolution axis; compare slopes, not levels."""
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    # (series, style, name, value labels below the points, name label offset in points)
    lines = (
        (SUBSET_LINE, NEUTRAL, "Subset, untuned recipe\n(~2,000 classes, 29 min/arm, holdout)", False, (16, 16)),
        (CODABENCH, RECIPE, "Tuned recipe, all data\n(144 min, CodaBench val)", True, (16, -4)),
        ({x: v[0] for x, v in FULL_FT.items()}, ADAPT,
         "Tuned recipe, equal-compute screen\n(Van, 10 photos/class, holdout)", False, (16, 0)),
    )
    for series, (fill, edge), name, below, offset in lines:
        xs, accs = list(series), list(series.values())
        ax.plot(xs, accs, color=edge, lw=2, zorder=2)
        ax.scatter(xs, accs, s=80, color=fill, edgecolors=edge, linewidths=1.5, zorder=3)
        for x, acc in series.items():
            ax.annotate(f"{acc:.1f}", (x, acc), xytext=(0, -12 if below else 10), textcoords="offset points",
                        ha="center", va="top" if below else "bottom", color=INK, fontsize=11)
        ax.annotate(name, (xs[-1], accs[-1]), xytext=offset, textcoords="offset points",
                    ha="left", va="center", color=edge, fontsize=12, fontweight="bold",
                    linespacing=1.15)

    ax.set_xticks([224, 336, 384, 448, 512], ["224", "336", "384", "448", "512"])
    ax.set_xlim(200, 690)
    ax.set_ylim(60, 88)
    ax.set_xlabel("Training resolution (px)", color=INK_2, labelpad=8)
    ax.set_ylabel("Top-1 (%)", color=INK_2, labelpad=8)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    style(ax)
    fig.text(0.01, 0.025, "All full fine-tuning of augreg2 ViT-B. Each line is its own experiment with its own data "
             "and evaluation set:\ncompare how steeply each line rises, not their heights. Single runs; "
             "384 / 448 CodaBench runs add speed-only options.", color=INK_2, fontsize=10, linespacing=1.3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
    plt.close(fig)


def single_line_figure(out: Path, series, epochs, color, ylim, title, footnote, note=None):
    """One experiment per chart, for the main slide: the gain (CodaBench) or the plateau (screen)."""
    fill, edge = color
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=PAGE)
    ax.set_facecolor(PAGE)
    xs, accs = list(series), list(series.values())
    ax.plot(xs, accs, color=edge, lw=2, zorder=2)
    ax.scatter(xs, accs, s=90, color=fill, edgecolors=edge, linewidths=1.5, zorder=3)
    for x, acc in series.items():
        ax.annotate(f"{acc:.1f}", (x, acc), xytext=(0, 12), textcoords="offset points",
                    ha="center", va="bottom", color=INK, fontsize=13, fontweight="bold")
        if epochs:
            ax.annotate(f"{epochs[x]} epochs", (x, acc), xytext=(0, -14), textcoords="offset points",
                        ha="center", va="top", color=INK_2, fontsize=11)
    if note:
        x, y, text = note
        ax.text(x, y, text, ha="center", va="top", color=edge, fontsize=12, fontweight="bold")

    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlim(xs[0] - 30, xs[-1] + 30)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", color=INK, fontsize=14, pad=14)
    ax.set_xlabel("Training resolution (px)", color=INK_2, labelpad=8)
    ax.set_ylabel("Top-1 (%)", color=INK_2, labelpad=8)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    style(ax)
    fig.text(0.01, 0.025, footnote, color=INK_2, fontsize=10, linespacing=1.3)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=PAGE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 13,
                         "text.color": INK, "xtick.color": INK_2, "ytick.color": INK})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    subset_figure(out / "resolution_subset")
    equal_compute_figure(out / "resolution_equal_compute")
    combined_figure(out / "resolution_combined")
    single_line_figure(out / "resolution_codabench", CODABENCH, None, RECIPE, (74, 85),
                       "Full FT, tuned recipe: CodaBench val",
                       "All 45 photos/class, 144 min on an A100 per run, plain prediction. 224 / 384 / 448 Van, "
                       "336 Anastasia;\nsame recipe, only the resolution changes (384 / 448 add speed-only options). "
                       "Single runs.")
    single_line_figure(out / "resolution_plateau", {x: v[0] for x, v in FULL_FT.items()},
                       {x: v[1] for x, v in FULL_FT.items()}, ADAPT, (62, 66.2),
                       "Same compute per run: higher resolution, fewer epochs",
                       "Van's equal-compute screen: full FT, tuned recipe, 10 training photos/class, "
                       "5 holdout photos/class,\ncrop 0.9. Every run gets the same GPU budget "
                       "(0.48 × 48 GB-hours).",
                       note=(480, 65.9, "+0.2: plateau"))


if __name__ == "__main__":
    main()
