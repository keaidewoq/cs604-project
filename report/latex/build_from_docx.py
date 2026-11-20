"""Convert the completed team report DOCX into the ICLR LaTeX body."""

from pathlib import Path
import re

import pypandoc


HERE = Path(__file__).resolve().parent
DOCX = HERE.parent / "CS604 Final Report (Team 08).docx"
RAW = HERE / "generated_word.tex"
OUT = HERE / "content.tex"


def conference_table(caption: str, label: str, spec: str, rows: str, *, wide: bool = True) -> str:
    env = "table*" if wide else "table"
    size = r"\scriptsize" if label == "tab:peft" else r"\small"
    width = r"\textwidth" if wide else r"\columnwidth"
    return rf"""
\begin{{{env}}}[t]
\caption{{{caption}}}
\label{{{label}}}
\centering{size}
\begin{{tabularx}}{{{width}}}{{@{{}}{spec}@{{}}}}
\toprule
{rows}
\bottomrule
\end{{tabularx}}
\end{{{env}}}
"""


TABLES = [
    conference_table(
        "Comparison of the final systems on the validation set (448 px, 144 training minutes on one A100). "
        "Latency is batch 1; peak VRAM uses physical batch 64.",
        "tab:final",
        "Xcc",
        r"""Metric & Full FT & AF + LoRA (PEFT) \\
\midrule
Validation top-1 (\%), no extensions / +zoom / +zoom+kNN & 82.5 / 83.22 / 84.07 & 81.2 / 82.2 / 83.08 \\
Trainable share of backbone & 100\% & 4.94\% \\
Peak training VRAM (GiB) & 17.80 & 14.66 \\
Epochs in 144 min & 5.5 & 5.5 \\
Latency, one pass (ms) & 8.47 & 11.27 (13.45 unmerged) \\
Latency with zoom (ms) & 23.43 & 29.88 \\""",
    ),
    conference_table(
        "Full fine-tuning ablations on the 10,000-class holdout at 224 px and 3 epochs unless stated.",
        "tab:ft-ablation",
        "Xccc",
        r"""Setting & Top-1 (\%) & Change & Compared with \\
\midrule
Pretrained features, prototype head, no training & 43.90 & -- & -- \\
Control: LP-FT head at backbone LR & 69.83 & -- & -- \\
+ head LR $10\times$ (LP-FT) & 71.37 & +1.54 & Control \\
No layer decay, same top LR & 63.12 & $-6.71$ & Control \\
Prototype head instead of LP-FT & 72.98 & +1.61 & LP-FT, head LR $10\times$ \\
Constant LR after warm-up & 63.99 & $-8.99$ & Prototype head \\
6 / 12 epochs instead of 3 & 73.97 / 74.49 & +0.99 / +0.52 & 3 / 6 epochs \\
336 px instead of 224 px & 78.39 & +5.41 & 224 px \\""",
    ),
    conference_table(
        "PEFT methods on validation at 336 px with no prediction-time extensions and a 144-minute budget. "
        "Small differences should be read as ties because recipes and physical batches differ. "
        "Parentheses show latency after merging.",
        "tab:peft",
        "Xcccc",
        r"""Method & Backbone trained & Top-1 (\%) & VRAM (GiB) & Latency (ms) \\
\midrule
Last-block attention & 2.75\% & 67.93 & 1.43 & 7.76 \\
Prefix, length 50 & 1.07\% & 72.1 & 7.97 & 9.64 \\
BitFit, all biases & 0.12\% & 73.4 & 7.01 & 7.40 \\
Serial adapter w96 + extras & 4.68\% & 77.6 & 15.54 & 11.60 \\
GPS & 4.63\% & 77.9 & 20.37 & 7.13 \\
AdaptFormer w192 & 4.13\% & 77.9 & 14.53 & 10.28 \\
DoRA r24 + extras & 4.75\% & 78.3 & 28.06 & 23--24 (7.71) \\
LoRA r24 & 4.12\% & 78.4 & 19.30 & 15.37 (7.78) \\
LoRA r24 + extras & 4.65\% & 78.5 & 19.40 & 14.2--15.0 (7.68--8.35) \\""",
    ),
    conference_table(
        "Validation versus test accuracy for full fine-tuning at 448 px.",
        "tab:test",
        "Xccc",
        r""" & Validation & Test & Change \\
\midrule
Zoom & 83.22 & 82.91 & $-0.31$ \\
Zoom + kNN & 84.07 & 83.84 & $-0.23$ \\
kNN gain & +0.85 & +0.93 & -- \\""",
        wide=False,
    ),
    conference_table(
        "Training and holdout accuracy per epoch.",
        "tab:train-holdout",
        "Xccc",
        r""" & Epoch 1 & Epoch 2 & Epoch 3 \\
\midrule
AdaptFormer, training & 82.91 & 92.48 & 94.97 \\
AdaptFormer, holdout & 70.26 & 76.06 & 77.45 \\
AdaptFormer, gap & 12.66 & 16.41 & 17.52 \\
Full FT, training & 78.17 & 91.78 & 95.46 \\
Full FT, holdout & 66.55 & 76.00 & 78.81 \\
Full FT, gap & 11.62 & 15.78 & 16.65 \\""",
    ),
    conference_table(
        "Equal-cost second views for full fine-tuning (336 px holdout; baseline 78.80\\%).",
        "tab:second-view",
        "Xr",
        r"""Second view & Gain (pp) \\
\midrule
Horizontal flip & +0.17 \\
Centre crop & +0.63 \\
Activation-guided zoom & +0.79 \\""",
        wide=False,
    ),
    conference_table(
        "kNN scoring variants on the holdout.",
        "tab:knn",
        "lXcc",
        r"""Variant & Query comparison & Gain (pp) & On top of zoom \\
\midrule
kNN, mean of 3 & Mean cosine to three closest class images & +1.32 & +1.23 \\
kNN, best match & Cosine to closest image & +1.18 & +1.02 \\
Weighted vote & 50 nearest images of any class & +1.14 & +1.03 \\
3 k-means centres & Best cosine to three class centres & +0.70 & +0.69 \\
Class mean & Cosine to average class feature & +0.55 & +0.49 \\
kNN, mean of all & Mean cosine to all class images & +0.13 & +0.13 \\""",
    ),
]


FIGURES = [
    ("Pipeline for the full fine-tuning and PEFT routes. Dots mark our changes; shaded boxes are prediction-time steps.", "fig:pipeline"),
    ("Epoch-mean training loss of the final systems at 448 px. The dashed line marks the 144-minute limit (about 5.5 epochs).", "fig:loss"),
    ("Resolution at equal compute. Full fine-tuning flattens after 448 px; AdaptFormer keeps improving to 448 px.", "fig:resolution"),
    ("Learning-rate sweep for full fine-tuning with layer-wise decay 0.75.", "fig:lr-sweep"),
    ("Full fine-tuning versus AdaptFormer + LoRA across accuracy, memory, parameters, time and latency.", "fig:summary"),
    ("Top-k holdout accuracy: the correct class is usually on the short list.", "fig:topk"),
    ("Zoom and kNN gains on the all-class holdout.", "fig:zoom-knn"),
    ("Examples where attention-guided zoom changed the prediction.", "fig:zoom-examples"),
]


CITATIONS = {
    "(Dosovitskiy et al., 2021)": r"\citep{dosovitskiy2021vit}",
    "(Kumar et al., 2022)": r"\citep{kumar2022lpft}",
    "(Steiner et al., 2022)": r"\citep{steiner2022augreg}",
    "(Qi et al., 2018)": r"\citep{qi2018imprinting}",
    "(Wang et al., 2018)": r"\citep{wang2018cosface}",
    "(He et al., 2022)": r"\citep{he2022mae}",
    "(Ben Zaken et al., 2022)": r"\citep{zaken2022bitfit}",
    "(Zhang et al., 2024)": r"\citep{zhang2024gps}",
    "(Lee et al., 2023)": r"\citep{lee2023surgical}",
    r"(Li \& Liang, 2021; Jia et al., 2022)": r"\citep{li2021prefix,jia2022vpt}",
    "(Houlsby et al., 2019)": r"\citep{houlsby2019adapter}",
    "(Chen et al., 2022)": r"\citep{chen2022adaptformer}",
    "(Hu et al., 2022)": r"\citep{hu2022lora}",
    "(Liu et al., 2024)": r"\citep{liu2024dora}",
    "(Zhang et al., 2023)": r"\citep{zhang2023adalora}",
    "(Fu et al., 2017)": r"\citep{fu2017racnn}",
    "(Wu et al., 2018)": r"\citep{wu2018instance}",
    "(Mensink et al., 2013)": r"\citep{mensink2013ncm}",
    "(Khandelwal et al., 2020)": r"\citep{khandelwal2020knnlm}",
    "(Zhou et al., 2016)": r"\citep{zhou2016cam}",
    "(Wightman, 2019)": r"\citep{rw2019timm}",
    "(doem1997, 2026)": r"\citep{cs604data}",
}


def build() -> None:
    pypandoc.convert_file(
        str(DOCX),
        "latex",
        outputfile=str(RAW),
        extra_args=[f"--extract-media={HERE / 'media'}", "--wrap=none"],
    )
    source = RAW.read_text(encoding="utf-8")
    match = re.search(
        r"\\subsection\{Abstract\}\\label\{abstract\}\s*(.*?)\n\n"
        r"\\subsection\{1\. Introduction\}\\label\{introduction\}",
        source,
        re.S,
    )
    if not match:
        raise RuntimeError("Could not locate the abstract and introduction")

    text = (
        "\\begin{abstract}\n"
        + match.group(1).strip()
        + "\n\\end{abstract}\n\n\\section{Introduction}\\label{introduction}"
        + source[match.end() :]
    )
    text = re.sub(
        r"\\subsection\{([2-6])\. ([^}]*)\}",
        lambda m: "\\section{" + m.group(2) + "}",
        text,
    )
    text = re.sub(
        r"\\subsubsection\{(?:3|5)\.\d+ ([^}]*)\}",
        lambda m: "\\subsection{" + m.group(1) + "}",
        text,
    )
    text = text.replace(
        r"\subsection{\texorpdfstring{Appendix }{Appendix }}\label{appendix}",
        "\\appendix\n\\section{Appendix}\\label{appendix}",
    ).replace(
        r"\subsubsection{Appendix figures and tables}\label{appendix-figures-and-tables}",
        r"\subsection{Appendix figures and tables}\label{appendix-figures-and-tables}",
    )

    # Captions are regenerated inside proper floats below.
    text = "\n".join(
        line
        for line in text.splitlines()
        if not (
            line.startswith(r"\emph{\textbf{Table")
            or line.startswith(r"\textbf{Table ")
            or line.startswith(r"\textbf{Figure ")
            or line.startswith(r"\emph{Figure ")
        )
    )

    table_pattern = re.compile(
        r"\{\\def\\LTcaptype\{none\}.*?\\end\{longtable\}\s*\}", re.S
    )
    blocks = table_pattern.findall(text)
    if len(blocks) != len(TABLES):
        raise RuntimeError(f"Expected {len(TABLES)} tables, found {len(blocks)}")
    for block, replacement in zip(blocks, TABLES):
        text = text.replace(block, replacement, 1)

    for index, (caption, label) in enumerate(FIGURES, 1):
        pattern = re.compile(
            rf"\\includegraphics\[[^\]]*\]\{{[^}}]*media/media/image{index}\.png\}}"
        )
        replacement = (
            "\\begin{figure*}[t]\n"
            "\\centering\n"
            f"\\includegraphics[width=0.82\\textwidth]{{media/media/image{index}.png}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\end{figure*}"
        )
        text, count = pattern.subn(lambda _: replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not replace figure {index}")

    equation = r"""\begin{equation}
s(c)=\log p(c\mid x)+\beta\frac{1}{m}\sum_{i\in\mathrm{top}\text{-}m}
\cos\!\left(f(x),f(x_i^c)\right).
\end{equation}
Here $K=5$, $m=3$ and $\beta=20$."""
    text = re.sub(
        r"s\(c\)=.*?\n\nK = 5, m = 3, β = 20",
        lambda _: equation,
        text,
        flags=re.S,
    )

    grouped = (
        r"(e.g., Hu et al., 2022; Jia et al., 2022; Chen et al., 2022)",
        r"\citep[e.g.,][]{hu2022lora,jia2022vpt,chen2022adaptformer}",
    )
    text = text.replace(*grouped)
    for plain, latex in CITATIONS.items():
        text = text.replace(plain, latex)

    text = (
        text.replace("×", r"$\times$")
        .replace("β", r"$\beta$")
        .replace("→", r"$\rightarrow$")
        .replace("−", "-")
    )
    text = text.replace(
        "\\appendix",
        "\\bibliography{refs}\n"
        "\\bibliographystyle{iclr2026_conference}\n\n"
        "\\appendix",
        1,
    )
    OUT.write_text(text.strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
    print(f"Wrote {OUT}")
