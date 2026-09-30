# Experiment record

The machine-readable source is `outputs/experiment_index.csv`. For each
leaderboard candidate, copy the following information into this file:

- Run name and Git commit:
- Method and motivation:
- Changed variable:
- Internal top-1 accuracy:
- Public top-1 accuracy:
- Total/trainable/backbone/classifier parameters:
- GPU, peak VRAM, wall time, and normalized GPU-hours:
- Inference milliseconds per image and batch size:
- Checkpoint and prediction file:
- Conclusion and next action:

## Completed

- Prototype starting-point validation, 2026-09-30: **43.900%** internal top-1
  (21,950 / 50,000) using the original holdout; **zero optimizer updates**.
  Config: `configs/experiments/prototype_validate_fp32_head.yaml`.
  Checkpoints: `C:\Users\Van Le\cs604-project\cs604-outputs\prototype-validate-fp32-head\baseline.pt`
  and `best.pt` in the same directory. No predictions or public score produced.
  Base commit: `010cf39aa28aef6987cdadaa15450e467b33f735`; dirty worktree on
  `prototype-baseline` (implementation was uncommitted when this run executed;
  it has since been committed with these results).
  Total parameters: 93,478,657; classifier/trainable-eligible: 7,680,001 (8.2158%);
  trainable backbone: 0. No parameters were updated in this evaluation-only run.
  RTX 5080 Laptop GPU, 15.92 GiB device capacity; peak allocated 1.0353 GiB;
  logged elapsed 351.9307 s, GPU-hours 0.0977585, normalized 48-GiB hours 0.0324232.
  Evaluation batch size 128, throughput 142.6225 images/s; no standalone prediction
  latency measurement. Source: the run's `run.json`, `baseline.json`, and `metrics.csv`.
  The earlier bf16-head attempt (`prototype-validate`) measured 43.768% and failed
  the unchanged 0.10-percentage-point tolerance. Using float32 final class scores
  with the same bf16 backbone reproduced the 43.90% reference. The two attempts
  consumed 708.4413 logged seconds total, excluding setup and original prototype
  construction. Both attempts are retained. Next: the separate head-only screen.

- Prototype initialization, 2026-09-29: average pretrained feature of 40 training images per class. Internal top-1 **43.90%** on the fixed 5-image holdout, with no training. File: `cs604-outputs/prototypes/class_prototypes.pt`. This matches the course zero-shot reference of 43.7%.

## Required comparisons

- Frozen backbone with trained linear head
- Full fine-tuning
- Selected final transformer blocks
- LoRA ranks 4, 8, and 16
- Frozen prototype head with LoRA
- Winning method with and without each final loss/augmentation change

Do not reconstruct these values at report time. `run.json`, `metrics.csv`, and
the `.inference.json` beside each prediction are the evidence for the report.
