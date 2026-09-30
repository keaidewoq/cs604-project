# Progress log

Newest session at the top. Each entry records what ran, which files went in and out, the result, and the decision. Training numbers still live in each output folder's `run.json`, `metrics.csv`, and `config.json`.

## 2026-09-29 — pipeline, full fine-tune, prototypes

### What changed

Built the experiment pipeline in `cs604-group-project`: a fixed internal holdout, training and evaluation, LoRA / frozen-head / last-block modes, prototype initialization, prediction files, and one YAML per experiment under `configs/experiments/`.

### Runs

**Throughput check.** `scripts/train.py` with `configs/experiments/benchmark.yaml`.

- Input: `C:\Users\Van Le\cs604-project\cs604-course-data`, 100 optimizer steps, 5 training images per class.
- Output: `C:\Users\Van Le\cs604-project\cs604-outputs\benchmark\`
- Result: about 23 images/second after warm-up on the RTX 5080 Laptop GPU. No accuracy was expected from this short probe.

**Full fine-tune.** `scripts/train.py` with `configs/experiments/full_baseline.yaml`.

- Input: the same dataset. All 40 training images per class. The classifier started from random weights.
- Output: `C:\Users\Van Le\cs604-project\cs604-outputs\full-baseline\` (`best.pt`, `last.pt`, `metrics.csv`, `run.json`)
- Result: internal top-1 **0.93%** after 12,500 steps and 4.45 hours. Peak allocated memory was 1.82 GB.

**Prototype initialization.** `scripts/build_prototypes.py` with `configs/experiments/prototype_init.yaml`.

- Input: the same dataset. Each class prototype is the average pretrained feature of its 40 training images. The 5 held-out images per class were not included in the average.
- Output: `C:\Users\Van Le\cs604-project\cs604-outputs\prototypes\class_prototypes.pt` and `class_prototypes.json`
- Result: internal top-1 **43.90%** with no training, in 1.35 hours. The course zero-shot reference is 43.7%.

### Decision

A random 10,000-class head does not learn in one full fine-tune. Later training runs should start from `class_prototypes.pt`. The prepared config for training only that head is `configs/experiments/prototype_head.yaml`. No leaderboard file has been submitted.
