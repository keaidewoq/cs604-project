# Experiment record

The machine-readable source is `report/experiment_index.csv`. For each
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

- Prototype LoRA rank-8 screen, 2026-09-30: internal top-1 **51.094%** (25,547 / 50,000) after two epochs, up from a remeasured start of **43.900%** (21,950 / 50,000). Gain versus the 43.90% prototype baseline: **+7.194 percentage points** (3,597 images). Epoch 1 was 50.508% (25,254 / 50,000); epoch 2 added 0.586 points. No public score.
  Config: `configs/experiments/prototype_lora_r8_fp32_screen.yaml`. Recorded commit: `910b2ff4d3cfcf89485243550fd429277d50aa95` on `prototype-baseline` (`git_dirty: true` because this config was uncommitted during the run).
  Method: frozen 40-image cosine prototypes, float32 class scores, LoRA rank 8 on attention `qkv` only (`train_head: false`). Ten images per class for the optimizer, so this is not a 10-shot study.
  Changed variable: 6,250 AdamW updates of the LoRA adapters (lr 0.0005, weight decay 0.01, label smoothing 0, batch 32, 2 epochs).
  Checkpoints: `C:\Users\Van Le\cs604-project\cs604-outputs\prototype-lora-r8-fp32-screen\baseline.pt` (step 0, accuracy 0.439) and `best.pt` / `last.pt` in the same directory (step 6250, accuracy 0.51094). No prediction file.
  Total parameters: 93,773,569; trainable LoRA: 294,912 (0.3145%); trainable classifier: 0.
  RTX 5080 Laptop GPU, 15.92 GiB device capacity; peak allocated 2.1347 GiB; logged elapsed 7351.8314 s (2h 2m 32s), GPU-hours 2.0421754, normalized 48-GiB hours 0.6773215.
  Initial holdout throughput 69.4753 images/s; training about 42 images/s in the second epoch; final holdout throughput 91.6021 images/s at evaluation batch size 64. Source: that run's `run.json`, `baseline.json`, and `metrics.csv`.
  The step-zero gate passed (`accuracy_delta` 0.0). Both epochs beat the start, and epoch 2 beat epoch 1, so `best.pt` is the epoch-2 model. This is above the head-only screen (45.858%) and under the 5% parameter budget. The original prototypes and earlier checkpoints were left in place.

- Prototype head-only screen, 2026-09-30: internal top-1 **45.858%** (22,929 / 50,000) after one epoch, up from a remeasured start of **43.900%** (21,950 / 50,000). Gain versus the 43.90% prototype baseline: **+1.958 percentage points** (979 images). No public score.
  Config: `configs/experiments/prototype_head_screen.yaml`. Commit: `929f8f361007a3ba90e08a21d6cd2b7592f12c08` on `prototype-baseline` (`git_dirty: false`).
  Method: frozen ViT-B/16 backbone, cosine head trained from the validated prototype checkpoint, with `cosine_head_fp32: true`. Five images per class for the optimizer; the initializer itself used 40 images per class, so this is not a five-shot result. The head is a diagnostic baseline, not the under-5% adaptation method.
  Changed variable: 782 AdamW updates of the classifier only (lr 0.001, weight decay 0, label smoothing 0, batch 64).
  Checkpoints: `C:\Users\Van Le\cs604-project\cs604-outputs\prototype-head-screen\baseline.pt` (step 0, accuracy 0.439) and `best.pt` / `last.pt` in the same directory (step 782, accuracy 0.45858). No prediction file.
  Total parameters: 93,478,657; trainable classifier: 7,680,001 (8.2158%); trainable backbone: 0.
  RTX 5080 Laptop GPU, 15.92 GiB device capacity; peak allocated 1.1328 GiB; logged elapsed 1606.8395 s (26m 47s), GPU-hours 0.4463443, normalized 48-GiB hours 0.1480375.
  Initial holdout throughput 157.8886 images/s; training about 80 images/s after warmup; final holdout throughput 79.1746 images/s at evaluation batch size 128. Source: that run's `run.json`, `baseline.json`, and `metrics.csv`.
  The step-zero gate passed (`accuracy_delta` 0.0). Training improved the fixed 50,000-image holdout, so `best.pt` is the trained head. `baseline.pt` and the earlier `prototype-validate-fp32-head` checkpoints were left in place. Next training run needs a new experiment YAML and output directory.

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
  construction. Both attempts are retained. The follow-up head-only screen is recorded above.

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
