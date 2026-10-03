# Prototype baseline: agent handoff

Updated: 2026-10-03. Read this together with `AGENTS.md` before continuing.
"Where to work" is current; the sections after it are the 2026-10-01 history.

## Current status (2026-10-03)

**Rules changed.** The Prof confirmed: the classifier head must be trained and
is not counted in the 5% limit, and any ViT of comparable size may be the
backbone. See "Course rules" in `AGENTS.md`. Every earlier LoRA result used a
frozen head and is now the frozen-head ablation.

**Latest runs (frozen head, rank-8 `qkv` LoRA, 40 images/class, Colab):**

| Run | GPU | Epochs | Normalized 48 GB-h | Internal top-1 |
| --- | --- | --- | --- | --- |
| `cs604-outputs/prototype_lora_r8_fp32_40` | RTX PRO 6000, 95 GB | 1 | 0.191 | 51.826% |
| `cs604-outputs/prototype_lora_r8_fp32_budget_40` | A100 80 GB | 6.7 (budget stop) | 1.901 | 52.924% |

The budget run (commit `8cb4732`) levelled off: per-epoch holdout accuracy was
49.74, 50.92, 51.77, 52.72, 52.70, 52.92, 52.91%. Ten times the budget bought
+1.1 points.

**Diagnosis** (evaluation-only, on the budget run's `best.pt`; script and
results in `cs604-outputs/prototype_lora_r8_fp32_budget_40/analysis/`):

| | Holdout (50,000) | Train sample (2/class, eval transform) |
| --- | --- | --- |
| Top-1 / top-5 | 52.92% / 78.24% | 61.55% / 83.61% |
| Mean cosine to own class prototype | 0.582 | 0.597 |
| Cross-entropy at the fixed scale 20 | 2.686 | 2.440 |
| Lowest cross-entropy over scales tried | 2.184 at scale 40 | 1.799 at scale 40 |

- Underfitting, not overfitting: 61.6% on images it trained on for 6.7 epochs.
- The frozen cosine temperature (20) is too low; about 40 fits better.
- Features stay far from the frozen prototypes (cosine 0.58).
Conclusion: training length is not the bottleneck; the frozen head and small
trainable capacity are. This matches the Prof's guidance to train the head.

**Next experiments (in order; details in `report/INAT2021_STRATEGY.md`):**

1. Code: separate head learning rate (`head_lr`), configurable train crop
   scale, then the trained-head screen config.
2. Trained-head screen on the current backbone: rank-8 `qkv` LoRA plus
   prototype-initialized trainable cosine head (temperature trains with it),
   10 images/class, against `prototype_lora_r8_fp32_screen` (51.094%).
3. Backbone screen without training: prototype accuracy on the fixed holdout
   for each candidate ViT-B. Run on Colab; the laptop evaluated at only about
   28 images/s (about 30 min per 50,000 images) on 2026-10-03.
4. More backbone capacity within 5% (LoRA on all linear layers or higher rank;
   AdaptFormer), then a prototype-initialized full fine-tune.
5. Full-budget runs only for finalists. Prefer a 40 GB A100 to an 80 GB one:
   the budget is GB-hours, so 40 GB gets 2.4 h versus 1.2 h.

Time budget tooling (`budget_48gb_hours`, elapsed time carried through resume)
and the Colab watchdog plus Drive flush are documented in `WORKFLOW.md`.

## Where to work

- Active worktree: `C:\Users\Van Le\cs604-project\cs604-prototype-baseline`
- Branch: `prototype-baseline`
- Base commit: `010cf39aa28aef6987cdadaa15450e467b33f735`
- Original checkout: `C:\Users\Van Le\cs604-project\cs604-group-project` (unchanged).
- Dataset and outputs remain shared siblings: `cs604-course-data` and `cs604-outputs`.
- Work on this worktree when switching between Codex, Claude Code, and Cursor.
  The implementation, head-screen write-up, and rank-8 LoRA screen config
  and write-up are on this branch. The LoRA training run recorded
  `git_dirty: true` at `910b2ff4d3cfcf89485243550fd429277d50aa95`, because its
  config was not committed before that run. Use `git log -1` for the latest
  commit. Do not switch back to the original checkout and assume these changes
  are present there. The branch is pushed to `origin` (Colab checks out its
  tip) but not merged into `main`.
- The app's managed-worktree tool could not recognize the nested repository;
  this is a normal Git worktree created with `git worktree add`.

## Current status

Evaluation-only validation is **complete**: 43.900% (21,950 / 50,000), with zero
optimizer updates. The validated checkpoint is
`cs604-outputs/prototype-validate-fp32-head/best.pt`.

The head-only screen is also **complete**. It remeasured that same start at
43.900%, then scored **45.858%** (22,929 / 50,000) after 782 classifier updates:
**+1.958 percentage points** versus the 43.90% baseline. Output:
`cs604-outputs/prototype-head-screen`. `best.pt` there is the trained checkpoint;
`baseline.pt` is still the zero-update model. Do not rerun into that directory.

The rank-8 LoRA screen is **complete**. Step zero was again 43.900%. After 6,250
adapter updates the holdout scored **51.094%** (25,547 / 50,000): **+7.194
percentage points** versus 43.90%, and above the head-only screen. Output:
`cs604-outputs/prototype-lora-r8-fp32-screen`. `best.pt` there is the epoch-2
checkpoint. Do not rerun into that directory.

The first attempt, `prototype-validate`, scored 43.768% and correctly failed the
unchanged 43.90% +/- 0.10 percentage-point gate. The original prototype script
scores normalized features in float32, while the training classifier previously
performed its final comparison inside bf16 autocast. Added `cosine_head_fp32: true`
to keep normalization and class scoring in float32, while retaining the bf16
backbone. The corrected run reproduced 43.900%. Both attempts are preserved;
do not overwrite them or use the failed attempt as the training initializer.

The code now:

1. Checks prototype split metadata (when loading prototypes), the exact saved
   internal-holdout file hash, and the expected 50,000 held-out images.
2. Saves `baseline.pt` before evaluation and before any optimizer update.
3. Evaluates through the same `build_from_config` / `evaluate` path used in training.
4. Requires top-1 accuracy within **0.10 percentage points** of 43.90%:
   acceptable range 43.80% through 44.00%, inclusive. This is an initial diagnostic
   tolerance for float32 prototype scoring versus bf16 cosine logits, not a
   measured confidence interval. Investigate discrepancies; do not relax the
   tolerance merely to make a failing run pass.
5. Records `baseline.json`, an `initial_val` row in `metrics.csv`, and `run.json`.
   Source checkpoint/prototype and holdout SHA-256 hashes are in `baseline.json`.
6. On success, initializes `best.pt` to the starting model. Only a strictly better
   held-out score can replace it during training. `baseline.pt` stays unchanged.
7. Refuses to overwrite an existing protected run on a fresh invocation. A failed
   accuracy check preserves the diagnostic baseline but does not produce `best.pt`.

The checkpoints contain model weights and config, so the existing `scripts/predict.py`
can load them. No prediction or leaderboard submission has been made.
Run metadata also records the source checkout and branch. Git reads trust only
that exact checkout for each command, avoiding worktree-ownership failures without
editing global Git configuration.

## Completed experiment: validate without training

This is the command used for the successful run. Do not rerun into its existing
output directory; the overwrite guard will reject it:

```powershell
Set-Location 'C:\Users\Van Le\cs604-project\cs604-prototype-baseline'
& 'C:\Users\Van Le\.venvs\cs604\Scripts\python.exe' -u scripts/train.py --config configs/experiments/prototype_validate_fp32_head.yaml
```

Input: the existing `cs604-outputs/prototypes/class_prototypes.pt`, built from
40 training images per class, and its original internal holdout. Only the 50,000
held-out images are forwarded through the model; no optimizer updates occur.
The configured five training images per class are used only to construct the
split and are not processed in this evaluation-only run.

Output: `C:\Users\Van Le\cs604-project\cs604-outputs\prototype-validate-fp32-head`.

Success checks:

- `run.json`: `status` is `evaluated`, `optimizer_steps` is 0.
- `baseline.json`: `passed` is true and `val_images` is 50000.
- `metrics.csv`: one `initial_val` row, epoch 0 / step 0, no training rows.
- `baseline.pt` and `best.pt` exist with the measured starting accuracy.

If validation fails, read `run.json` and any `baseline.json` before making changes.
Check image transforms, normalization, source weights, precision, and the split.
Use a new output directory in a separate experiment YAML for a rerun; preserve the
failed run for diagnosis. Never delete or overwrite the original prototypes.

## Completed experiment: small head-only screen

Ran `configs/experiments/prototype_head_screen.yaml` on 2026-09-30. The overwrite
guard will reject a fresh invocation into the same output directory. Do not rerun
it. An interrupted copy would have been resumed with `--resume` pointing at that
run's `last.pt`; this run finished, so it does not need a resume.

```powershell
Set-Location 'C:\Users\Van Le\cs604-project\cs604-prototype-baseline'
& 'C:\Users\Van Le\.venvs\cs604\Scripts\python.exe' scripts/train.py --config configs/experiments/prototype_head_screen.yaml
```

It loaded `prototype-validate-fp32-head/best.pt`, evaluated step zero again,
froze the backbone, and trained the cosine classifier on five images per class.
One epoch / 50,000 training images / batch 64 gave 782 optimizer updates.
LR was 0.001 with the existing warmup/cosine schedule; weight decay and label
smoothing were zero. It used the same 50,000-image holdout before and after training
and set `cosine_head_fp32: true`.

Output: `cs604-outputs/prototype-head-screen`.

Measured result, from that folder's `run.json`, `baseline.json`, and `metrics.csv`:

- Step zero: 43.900% (21,950 / 50,000), gate passed, `accuracy_delta` 0.0.
- After training: 45.858% (22,929 / 50,000), loss 2.6540017303466796.
- Comparison with the 43.90% baseline: +1.958 percentage points.
- `baseline.pt` remains step 0. `best.pt` and `last.pt` are step 782 because the
  trained score was strictly better.
- Logged elapsed 1606.8395 s, peak allocated 1.1328 GiB, 782 optimizer steps.
- Commit `929f8f361007a3ba90e08a21d6cd2b7592f12c08`, clean worktree.

This is a small optimization screen, **not a five-shot learning result**: its
prototype initialization already used 40 labeled images per class. The head has
about 8.22% of total model parameters and is a diagnostic baseline, not the final
under-5% adaptation method. The earlier validation checkpoints and
`prototypes/class_prototypes.pt` were not modified.

## Completed experiment: rank-8 LoRA screen

Ran `configs/experiments/prototype_lora_r8_fp32_screen.yaml` on 2026-09-30.
This is the prepared `prototype_lora_r8_screen.yaml` setup with
`cosine_head_fp32: true` and a step-zero accuracy gate. The original YAML was
left unchanged. Do not rerun into the new output directory.

Measured result, from `cs604-outputs/prototype-lora-r8-fp32-screen`:

- Step zero: 43.900% (21,950 / 50,000), gate passed, `accuracy_delta` 0.0.
- Epoch 1: 50.508% (25,254 / 50,000).
- Epoch 2: 51.094% (25,547 / 50,000), loss 2.8203868717956544.
- Comparison with the 43.90% baseline: +7.194 percentage points.
- Comparison with the head-only screen: 51.094% versus 45.858%.
- Trainable parameters: 294,912 (0.3145%), classifier frozen.
- `baseline.pt` remains step 0. `best.pt` and `last.pt` are step 6250.
- Logged elapsed 7351.8314 s, peak allocated 2.1347 GiB, 6,250 optimizer steps.
- Recorded commit `910b2ff4d3cfcf89485243550fd429277d50aa95`, dirty worktree.

The frozen head is still the original 40-image prototype file. The optimizer
saw ten images per class. `prototypes/class_prototypes.pt` and the earlier
checkpoints were not modified.

## Verification

```powershell
& 'C:\Users\Van Le\.venvs\cs604\Scripts\python.exe' -m unittest tests.test_pipeline tests.test_baseline
& 'C:\Users\Van Le\.venvs\cs604\Scripts\python.exe' scripts/train.py --config configs/smoke.yaml --synthetic --output-dir 'C:\Users\Van Le\cs604-project\cs604-outputs\smoke-prototype-baseline'
```

- 15 unit tests passed. Control-flow tests use tiny CPU models and scripted evaluation
  scores to test control flow; their printed 43.90% is **not** a real-data result.
- Tests cover zero updates, reloadable checkpoints, worse/better training,
  failed accuracy gates, overwrite refusal, split mismatch, and resume protection.
- Precision tests confirm full-precision scores stay distinct under outer autocast,
  gradients still flow, and checkpoint configuration restores the precision setting.
- The five-step synthetic GPU smoke test passed on the RTX 5080 Laptop GPU.
  Its outputs are in `cs604-outputs/smoke-prototype-baseline`. This checks the
  existing real-model training path, not accuracy on the course holdout.
- Real-data validation reproduced 43.900% in 351.93 logged seconds (about 5m 52s),
  with 1.0353 GiB peak allocated GPU memory. The logged duration starts after model
  and data setup. Both attempts together used 708.44 logged seconds; initialization
  remains separately accounted for in the original prototype-building run.
- The head-only screen then remeasured 43.900% at step zero and finished at
  45.858% after 782 updates. Logged elapsed 1606.8395 s, peak allocated 1.1328 GiB.
  Those figures are copied from `cs604-outputs/prototype-head-screen`.
- The rank-8 LoRA screen remeasured 43.900% at step zero, reached 50.508% after
  epoch 1, and finished at 51.094% after epoch 2. Logged elapsed 7351.8314 s,
  peak allocated 2.1347 GiB. Those figures are copied from
  `cs604-outputs/prototype-lora-r8-fp32-screen`.

After a real experiment, add the measured result and paths to `progress_log.md`
and `EXPERIMENTS.md`, and update the experiment index. Do not infer scores from
configurations or from these unit tests.

## Starting a fresh agent chat

Use the existing worktree on branch `prototype-baseline`. Read `AGENTS.md` and
this handoff. The head-only screen and the rank-8 LoRA screen have both been
run and recorded. Do not start another GPU job in
`cs604-outputs/prototype-head-screen` or
`cs604-outputs/prototype-lora-r8-fp32-screen`, and do not overwrite
`prototype-validate-fp32-head` or the original prototypes. A later training run
needs its own experiment YAML and output directory, and should start only when
the user requests that run. The prepared full-data LoRA finalist still omits
`cosine_head_fp32: true`.

Checkpoint files, raw run logs, and the generated experiment index remain local
under `cs604-outputs` and ignored `outputs/`; Git stores the implementation,
experiment configurations, tests, and results documentation, not the large models.
