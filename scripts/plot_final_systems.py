"""Slide figures for the two finals (numbers: report/final_report_draft.md sections 0 and 5, CodaBench).

  final_system  left: both finals at each prediction stage. right: what built Task 1's score, from
                the first scheduled full FT to the final (66.3, recipe +9.4, resolution +6.8, zoom, kNN).
  ft_vs_peft    who wins on each measure. Each row's bars share a scale that starts at zero, and
                rows have different units, so compare within a row only.

    python scripts/plot_final_systems.py --out_dir figures
writes <name>.png and <name>.svg for each.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from plot_full_ft_ablation import INK, INK_2, NEUTRAL, PAGE, RECIPE, RULE

FT_COLOUR, PEFT_COLOUR = "#3d3d3a", "#1d9e75"
STAGES = ["Single view", "+ zoom", "+ zoom + kNN"]
FT_SCORES = [82.5, 83.22, 84.07]
PEFT_SCORES = [81.2, 82.2, 83.08]
# (label, gain, or None for the starting score)
STEPS = [("Scheduled\nfull FT, 224", 66.3, None), ("Recipe", 9.4, RECIPE), ("Resolution\n224 → 448", 6.8, RECIPE),
         ("Zoom", 0.72, RECIPE), ("kNN", 0.85, RECIPE)]
# (measure, full FT value, PEFT value, full FT text, PEFT text, winner)
ROWS = [
    ("Accuracy", "CodaBench, zoom + kNN", 84.07, 83.08, "84.07", "83.08", "Full FT"),
    ("Trainable backbone", "share of parameters", 100, 4.94, "100%", "4.94%", "PEFT"),
    ("Peak VRAM", "during training", 17.80, 14.66, "17.80 GB", "14.66 GB", "PEFT"),
    ("Latency, one pass", "batch 1, PEFT with LoRA merged", 8.47, 11.27, "8.47 ms", "11.27 ms", "Full FT"),
    ("Latency with zoom", "batch 1, PEFT with LoRA merged", 23.43, 29.88, "23.43 ms", "29.88 ms", "Full FT"),
    ("Epochs in 144 minutes", "same training budget", 5.5, 5.5, "5.5", "5.5", "Tie"),
]


def style():
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 12,
                         "text.color": INK, "xtick.color": INK, "ytick.color": INK_2})


def clean(ax):
    ax.set_facecolor(PAGE)
    ax.tick_params(length=0, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=RULE, lw=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def stages_panel(ax):
    clean(ax)
    xs = range(len(STAGES))
    ax.plot(xs, FT_SCORES, "-o", color=FT_COLOUR, linewidth=2, markersize=9, markeredgecolor=PAGE, markeredgewidth=1.5, zorder=3)
    ax.plot(xs, PEFT_SCORES, "-s", color=PEFT_COLOUR, linewidth=2, markersize=9, markeredgecolor=PAGE, markeredgewidth=1.5, zorder=3)
    for x, ft, peft in zip(xs, FT_SCORES, PEFT_SCORES):
        ax.text(x, ft + 0.17, f"{ft:.2f}" if x else f"{ft:.1f}", ha="center", va="bottom", fontweight="bold")
        ax.text(x, peft - 0.17, f"{peft:.2f}" if x == 2 else f"{peft:.1f}", ha="center", va="top", fontweight="bold")
    ax.text(0.03, 84.15, "Full FT, 100%", color=INK, fontsize=12, va="center", ha="left")
    ax.plot(-0.12, 84.15, "o", color=FT_COLOUR, markersize=8)
    ax.text(1.22, 81.15, "AdaptFormer + LoRA, 4.94%", color=INK, fontsize=12, va="center", ha="center")
    ax.plot(0.47, 81.15, "s", color=PEFT_COLOUR, markersize=8)
    ax.set_xticks(list(xs), STAGES)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(80.5, 85.2)
    ax.set_yticks(range(81, 86))
    ax.set_ylabel("CodaBench top-1 (%)", color=INK_2, labelpad=8)
    ax.set_title("Same pipeline, 1.0 point apart at the end", loc="left", fontsize=13.5, fontweight="bold", pad=12)


def steps_panel(ax):
    clean(ax)
    bottom, width = 60, 0.62
    level = 0
    for x, (label, value, gain_style) in enumerate(STEPS):
        start = level if gain_style else bottom
        height = value if gain_style else value - bottom
        fill, edge = gain_style or NEUTRAL
        ax.add_patch(FancyBboxPatch((x - width / 2, start), width, height, boxstyle="round,pad=0,rounding_size=0.015",
                                    mutation_aspect=20, facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=3))
        level = start + height
        ax.text(x, level + 0.5, f"+{value:.2f}" if value < 1 else (f"+{value:.1f}" if gain_style else f"{value:.1f}"),
                ha="center", va="bottom", fontweight="bold")
        if x < len(STEPS) - 1:
            ax.plot([x + width / 2, x + 1 - width / 2], [level, level], color=INK_2, linewidth=0.7, zorder=2)
    ax.text(len(STEPS) - 1, level + 2.2, f"{level:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(STEPS)), [s[0] for s in STEPS], fontsize=11)
    ax.set_xlim(-0.55, len(STEPS) - 0.45)
    ax.set_ylim(bottom, 90)
    ax.set_yticks(range(60, 91, 10))
    ax.set_ylabel("CodaBench top-1 (%), axis starts at 60", color=INK_2, labelpad=8)
    ax.set_title("What built the Task 1 score", loc="left", fontsize=13.5, fontweight="bold", pad=12)


def final_system(out):
    style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.4), facecolor=PAGE, gridspec_kw={"width_ratios": [1, 1.05]})
    stages_panel(left)
    steps_panel(right)
    fig.tight_layout(w_pad=3, rect=(0, 0.085, 1, 1))
    fig.text(0.012, 0.018, "Both finals: 448 px, 144 training minutes, crop 0.9 with zoom. "
             "Recipe = layer decay, fitted schedule, prototype head. The resolution step's runs are not perfectly matched.\n"
             "Task 2's zoom and zoom + kNN scores come from separate prediction runs of the same checkpoint.",
             fontsize=9.5, color=INK_2, linespacing=1.3)
    save(fig, out)


def row_bar(ax, x, y, length, height, colour):
    ax.add_patch(FancyBboxPatch((x, y), max(length, 0.04), height, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=colour, edgecolor="none", zorder=3))


def ft_vs_peft(out):
    style()
    width, height = 12.0, 5.8
    fig = plt.figure(figsize=(width, height), facecolor=PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    bars_x, bars_w = 4.3, 4.5
    head = 5.45
    ax.text(0.25, head, "Measure", fontsize=11, color=INK_2, va="center")
    ax.plot(bars_x + 0.08, head, "o", color=FT_COLOUR, markersize=8)
    ax.text(bars_x + 0.22, head, "Full FT", fontsize=11, color=INK_2, va="center")
    ax.plot(bars_x + 1.15, head, "s", color=PEFT_COLOUR, markersize=8)
    ax.text(bars_x + 1.29, head, "AdaptFormer + LoRA", fontsize=11, color=INK_2, va="center")
    ax.text(10.75, head, "Winner", fontsize=11, color=INK_2, va="center")
    ax.plot([0.25, 11.75], [head - 0.2, head - 0.2], color=RULE, linewidth=0.8)

    row_h = 0.76
    y = head - 0.25
    for name, note, ft, peft, ft_text, peft_text, winner in ROWS:
        y -= row_h
        top = max(ft, peft)
        ax.text(0.25, y + row_h * 0.62, name, fontsize=13, fontweight="bold", va="center")
        ax.text(0.25, y + row_h * 0.28, note, fontsize=10, color=INK_2, va="center")
        for offset, value, text, colour in ((0.41, ft, ft_text, FT_COLOUR), (0.11, peft, peft_text, PEFT_COLOUR)):
            length = value / top * bars_w
            row_bar(ax, bars_x, y + offset, length, 0.24, colour)
            ax.text(bars_x + length + 0.1, y + offset + 0.12, text, fontsize=11.5, va="center")
        ax.text(10.75, y + row_h / 2, winner, fontsize=13, fontweight="bold" if winner != "Tie" else "normal", va="center",
                color=INK if winner != "Tie" else INK_2)
        ax.plot([0.25, 11.75], [y, y], color=RULE, linewidth=0.4)

    ax.text(0.25, 0.3, "Each row has its own scale starting at zero; compare within a row only. Latencies: A100, one session, "
            "per image. Before merging LoRA, PEFT is 13.45 ms and 33.78 ms.", fontsize=9.5, color=INK_2, va="center")
    save(fig, out)


def save(fig, out):
    fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAGE)
    fig.savefig(out.with_suffix(".svg"), facecolor=PAGE)
    plt.close(fig)
    print(f"wrote {out}.png and .svg")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_system(out_dir / "final_system")
    ft_vs_peft(out_dir / "ft_vs_peft")


if __name__ == "__main__":
    main()
