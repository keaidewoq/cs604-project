# Prototype baseline: agent handoff

Updated: 2026-09-30. Read this together with `AGENTS.md` before continuing.

## Where to work

- Active worktree: `C:\Users\Van Le\cs604-project\cs604-prototype-baseline`
- Branch: `prototype-baseline`
- Base commit: `010cf39aa28aef6987cdadaa15450e467b33f735`
- Original checkout: `C:\Users\Van Le\cs604-project\cs604-group-project` (unchanged).
- Dataset and outputs remain shared siblings: `cs604-course-data` and `cs604-outputs`.
- Work on this worktree when switching between Codex, Claude Code, and Cursor.
  The implementation and results documentation are committed on this branch;
  use `git log -1` for the commit ID. Do not switch back to the original checkout
  and assume these changes are present there. The branch has not been merged or pushed.
- The app's managed-worktree tool could not recognize the nested repository;
  this is a normal Git worktree created with `git worktree add`.

## Current status

Evaluation-only validation is **complete**: 43.900% (21,950 / 50,000), with zero
optimizer updates. The validated checkpoint is
`cs604-outputs/prototype-validate-fp32-head/best.pt`. The subsequent head-only
screen is prepared and points to that checkpoint, but has **not** been run.

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

## Subsequent experiment: small head-only screen

This is a separate real training run, not automatically started by validation:

```powershell
Set-Location 'C:\Users\Van Le\cs604-project\cs604-prototype-baseline'
& 'C:\Users\Van Le\.venvs\cs604\Scripts\python.exe' scripts/train.py --config configs/experiments/prototype_head_screen.yaml
```

It loads the validated `prototype-validate-fp32-head/best.pt`, evaluates step zero again,
freezes the backbone, and trains the cosine classifier on five images per class.
One epoch / 50,000 training images / batch 64 gives 782 optimizer updates.
LR is 0.001 with the existing warmup/cosine schedule; weight decay and label
smoothing are zero. It uses the same 50,000-image holdout before and after training
and sets `cosine_head_fp32: true` to retain the validated scoring behavior.

Output: `cs604-outputs/prototype-head-screen`. `best.pt` may legitimately remain
the zero-update model if training does not improve it; `last.pt` is the trained
model. Compare the actual recorded initial accuracy with final validation accuracy.
Resume an interrupted screen in its original output directory with `--resume`
pointing to its `last.pt`; do not resume the evaluation-only run.

This is a small optimization screen, **not a five-shot learning result**: its
prototype initialization already used 40 labeled images per class. The head has
about 8.22% of total model parameters and is a diagnostic baseline, not the final
under-5% adaptation method.

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

After a real experiment, add the measured result and paths to `progress_log.md`
and `EXPERIMENTS.md`, and update the experiment index. Do not infer scores from
configurations or from these unit tests.

## Starting a fresh agent chat

Use the existing worktree on branch `prototype-baseline`. Read `AGENTS.md` and
this handoff. The next task is to run `configs/experiments/prototype_head_screen.yaml`,
check the initial and final holdout scores, retain the best checkpoint, and record
the measured result. Start that training only when the user requests it.

Checkpoint files, raw run logs, and the generated experiment index remain local
under `cs604-outputs` and ignored `outputs/`; Git stores the implementation,
experiment configurations, tests, and results documentation, not the large models.
