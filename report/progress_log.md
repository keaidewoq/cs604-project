# Progress log

Newest session at the top. Each entry records what ran, which files went in and out, the result, and the decision. Training numbers still live in each output folder's `run.json`, `metrics.csv`, and `config.json`.

## 2026-09-30 — head-only screen completed

Worktree: `C:\Users\Van Le\cs604-project\cs604-prototype-baseline`, branch
`prototype-baseline`, commit `929f8f361007a3ba90e08a21d6cd2b7592f12c08`
(`git_dirty: false`).

**Run:** `configs/experiments/prototype_head_screen.yaml` through
`scripts/train.py` and the CUDA virtualenv. When this session checked the GPU,
that command was already the only training process, so no second job was started.

- Input: `C:\Users\Van Le\cs604-project\cs604-outputs\prototype-validate-fp32-head\best.pt`,
  the original prototypes, and the fixed 50,000-image internal holdout.
  The optimizer saw five training images per class for one epoch (782 updates).
  The backbone stayed frozen and class scores stayed in float32.
- Output: `C:\Users\Van Le\cs604-project\cs604-outputs\prototype-head-screen`

**Starting accuracy:** **43.900%**, or **21,950 / 50,000**, loss 3.501497850265503.
`baseline.json` records `passed: true`, `accuracy_delta: 0.0`, and
`optimizer_steps: 0`. This matches the 43.90% prototype baseline before any
head update. `baseline.pt` is that step-zero checkpoint and was not rewritten
after 21:27:49.

**Final holdout:** **45.858%**, or **22,929 / 50,000**, loss 2.6540017303466796,
at step 782. That is **+1.958 percentage points** versus 43.90% (979 additional
correct images). `run.json` status is `completed`, `best_internal_accuracy` is
0.45858, and `optimizer_steps` is 782. Because the trained score is strictly
higher, `best.pt` and `last.pt` both store step 782. Logged elapsed time is
**1606.8395 s** (26m 47s), including the initial evaluation. Peak allocated
memory is **1.1328 GiB**. GPU-hours 0.4463443; normalized 48-GiB hours 0.1480375.

Left unchanged: `prototype-validate-fp32-head/best.pt` (SHA-256 still
`57214311e11c71b21bc79e92f629a8a01d23d5aa53c8187ec6be9d632159e1b2`), that run's
`baseline.pt`, the failed `prototype-validate` checkpoint, and
`prototypes/class_prototypes.pt`. No leaderboard file was written.

Decision: one epoch of head-only training improves the fixed holdout. It remains
a diagnostic screen. Initialization used 40 labeled images per class, and the
trainable head is 8.2158% of the model, above the under-5% adaptation target.
Do not rerun this config into the same output directory.

## 2026-09-30 — starting-point validation completed

Worktree: `C:\Users\Van Le\cs604-project\cs604-prototype-baseline`, branch
`prototype-baseline`, base commit `010cf39aa28aef6987cdadaa15450e467b33f735`
with uncommitted implementation changes (`git_dirty: true`).

**First evaluation:** `configs/experiments/prototype_validate.yaml`, using the
existing prototypes and the exact original 50,000-image internal holdout.
Output: `C:\Users\Van Le\cs604-project\cs604-outputs\prototype-validate`.
Top-1 **43.768%**, zero optimizer updates, 356.51 logged seconds. The check failed
because the score differed from 43.90% by 0.132 percentage points, outside the
predeclared 0.10-point tolerance. The diagnostic checkpoint and logs were preserved;
no `best.pt` was created.

**Correction and second evaluation:** `configs/experiments/prototype_validate_fp32_head.yaml`.
The original prototype script uses float32 normalization and similarity scoring,
whereas the cosine classifier previously scored inside bf16 autocast. Added an
explicit `cosine_head_fp32` option that keeps only the final scoring in float32.
The backbone remains bf16; model weights, split, prototypes, and tolerance are unchanged.

Output: `C:\Users\Van Le\cs604-project\cs604-outputs\prototype-validate-fp32-head`.
Result: **43.900%**, or **21,950 / 50,000 correct**, with **zero optimizer updates**.
The accuracy gate passed. Logged run time: **351.93 seconds** (5m 52s), peak
allocated memory **1.0353 GiB**. Evaluation throughput: **142.62 images/second**.
This reproduces the original internal score; it is not a leaderboard result.

Saved `baseline.pt`, `best.pt`, `baseline.json`, `run.json`, `config.json`, and
`metrics.csv`. See `checkpoint_audit.json` for the post-run checkpoint checks.
Both attempts used the same prototype-file and holdout SHA-256 hashes.

Verification after the correction: **15 unit tests** and the **five-step synthetic
GPU smoke test** passed. The original failed evaluation is retained for diagnosis.

Decision: starting-point preservation and validation are complete. The prepared
head-only screen now loads `prototype-validate-fp32-head/best.pt` and retains
`cosine_head_fp32: true`. **The head-only training screen has not been started.**

## 2026-09-30 — prototype starting-point protection (implementation)

Worktree: `C:\Users\Van Le\cs604-project\cs604-prototype-baseline`, branch
`prototype-baseline`. The original checkout is unchanged. See `HANDOFF.md` for
commands and cross-agent continuation instructions.

Added evaluation-only `configs/experiments/prototype_validate.yaml` and the separate
`prototype_head_screen.yaml`. The pipeline saves `baseline.pt` before updates,
checks the original holdout and starting accuracy, and includes step zero in
best-checkpoint selection. Fresh protected runs cannot overwrite existing results.

Verification: 13 unit tests passed using tiny CPU models and mocked evaluation
scores. The five-step synthetic GPU smoke test passed on the RTX 5080 Laptop GPU,
writing to `C:\Users\Van Le\cs604-project\cs604-outputs\smoke-prototype-baseline`.
No real-data validation or training was launched, so **43.90% has not yet been
remeasured** by the new experiment.

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
