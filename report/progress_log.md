# Progress log

One short note per run. After CodaBench, also add a row to `experiment_index.csv`.

## Template

```

### YYYY-MM-DD — name — config
- run_dir:
- val acc (ours / CodaBench):
- trainable % / peak VRAM / GPU-hours:
- what changed:
- next:
```

### 2026-10-21 — Van — AdaptFormer + biases/norms + LoRA on qkv (laptop screen; Colab config ready)

- run_dir: none yet; Colab config `configs/van_adaptformer_lora_448_fast.yaml` (METHOD in `notebooks/van_colab_launcher.ipynb`)
- val acc (ours, laptop holdout 336 px, 40/5 split): 1 epoch: LoRA rank 8 73.54, rank 16 73.74 vs 72.97 without LoRA; **3 epochs: rank 16 77.33 vs 76.76 (+0.57)**.
- trainable %: 4.96% at 336 (4.94% at 448), head excluded; laptop throughput 9% lower, memory +0.5 GB.
- what changed: `adapter_lora: true` (opt-in, `method: van_adaptformer` only) wraps every qkv in the team's `LoRALinear` (rank 16, alpha 32) beside the adapters and the trainable biases/norms. Same logits as Van's laptop implementation on a trained checkpoint (max difference 0).
- next: Colab A100 run at 448, against 0.804 plain / 0.814 crop 0.9 + zoom.

### 2026-10-20 — Van — full FT 448: zoom from a crop 0.9 view (prediction only)

- run_dir: `CS604_RUNS/full_ft_20261017_122237` (same `last.pt`; nothing retrained)
- val acc (ours / CodaBench): **0.832 crop 0.9 + zoom** (ID 1004167, no balancing) vs 0.828 crop 0.75 + zoom (ID 995951): **+0.4**.
- what changed: `src.predict --crop_pct 0.9 --zoom` instead of `--crop_pct 0.75 --zoom`; the same change gave the AdaptFormer at 448 +0.5 (0.814).
- next: class-count balancing on top of this view (the balanced crop 0.75 + zoom file scored 0.845).

### 2026-10-19 — Van — AdaptFormer 448: zoom from a crop 0.9 view (prediction only)

- run_dir: `CS604_RUNS/van_adaptformer_20261018_073912` (same `last.pt`; nothing retrained)
- val acc (ours / CodaBench): **0.814 crop 0.9 + zoom** (ID 1002322, Task 2 best without balancing) vs 0.809 crop 0.75 + zoom (ID 1001880): **+0.5**.
- what changed: `src.predict --crop_pct 0.9 --zoom` instead of `--crop_pct 0.75 --zoom` (mass 0.7 unchanged).
- why: flip and zoom added less at 448, so Van re-tuned the views on the internal holdout with the 448 AdaptFormer screen model (eval only): crop 0.75 / 0.9 / 1.0 x zoom mass 0.5 / 0.7 / 0.9 x min box 8 / 11. Chosen on half the holdout, crop 0.9 + mass 0.7 beat crop 0.75 by +0.28 on the other half. The 0.75 crop helps the plain view but hurts the zoom at 448, where the zoom already enlarges the subject.
- next: the full FT at 448 may gain from the same crop change (not yet tested); class-count balancing is not yet applied to this file.

### 2026-10-19 — Van — class-count balancing at prediction (`src.predict --balance_per_class`)

- run_dir: no training; the Task 1 final `CS604_RUNS/full_ft_20261017_122237/last.pt`, re-predicted on Van's laptop (batch 32, 4 workers) with `--crop_pct 0.75 --zoom --balance_per_class 5` (`src/van_balance.py`, van-dev bb24049)
- val acc (ours / CodaBench): internal holdout (exactly 5 images/class, laptop checkpoints at 384): full FT plain 79.03 → 80.92%, plain + zoom 79.68 → 81.35%; AdaptFormer plain 77.54 → 79.25%, plain + zoom 78.31 → 79.82% / **CodaBench 0.845** (ID 1002123) vs 0.828 for the same views without balancing (ID 995951): **+1.7, as the holdout predicted. Team best.**
- what it is: val has exactly 5 images per class (test 10), but argmax predictions pile up on some classes (up to 12–13 images) and leave others empty. Sinkhorn gives each class a weight and rescales the weights (50 iterations) until every class's expected count is 5, then takes the argmax of the reweighted softmax (top-100 classes kept per image). On val it changed 3,096 of 50,000 predictions; afterwards no class is empty and the largest has 9. Forcing exactly 5 per class (min-cost matching) did worse on the holdout (80.24 / 80.74%) than the soft version.
- note: this uses the known per-class counts of the split (stated in the dataset README). To check with the Prof that this is allowed before relying on it for Phase 2.
- next: Phase 2 test with `--balance_per_class 10`; the AdaptFormer 448 final (Task 2) can get the same treatment.

### 2026-10-18 — Van — AdaptFormer + biases/LayerNorms at 448 px (`configs/van_adaptformer_bias_448_fast.yaml`)

- run_dir: `CS604_RUNS/van_adaptformer_20261018_073912` (A100 40 GB, `notebooks/van_colab_launcher.ipynb`, branch `van-dev` d65cbc3); prototypes `CS604_RUNS/prototypes_448.pt` (Qian's, all 45 images/class)
- val acc (ours / CodaBench): no internal holdout (all 45 images/class) / **0.804 plain** (ID 1000827), **0.806 crop 0.75 + flip** (ID 1000837), **0.809 crop 0.75 + zoom** (ID 1001880, Task 2 best). Against 384 (`van_adaptformer_20261016_142837`): +1.0 plain, +0.2 zoom.
- trainable % / peak VRAM / GPU-hours: 4.26% of the backbone excl. head (3,672,588; 11,352,589 incl. head) / 13.76 GB / 2.40 (144.0 training minutes, time limit reached)
- what changed vs `van_adaptformer_bias_384_fast.yaml`: `img_size` 384 → **448**, `prototypes_448.pt`, batch 128 x 2 → **64 x 4** (same effective 256). ~317 img/s, **6.1 epochs** (384: 8.6).
- why: Van's laptop screen at equal compute (10 images/class, crop 0.9): 336 / 384 / 448 = 59.63 / 61.18 / 62.72%, still +1.5 per step for AdaptFormer.
- note: the Colab zoom prediction (`zoom_crop075/`) stopped at 31,062 of 50,000 rows; do not submit it. The submitted zoom file was predicted on Van's laptop with batch 32 and 4 loader workers (zoom at 448 builds 896 px copies; on the laptop batch 64 x 12 workers crashed the same way, so memory is the likely cause on Colab too). Its first 31,062 rows agree with the partial Colab file on 99.5% of images.
- next: as for full FT, flip and zoom add less at higher resolution (+0.2 / +0.5 over plain at 448, vs +1.3 for zoom at 384). Van is checking whether the eval crop and zoom box need retuning at 448.

### 2026-10-17 — Van — full FT at 448 px (`configs/van_full_ft_448_fast.yaml`)

- run_dir: `CS604_RUNS/full_ft_20261017_122237` (A100 40 GB, `notebooks/van_colab_launcher.ipynb`, branch `van-dev` 2013466); prototypes `CS604_RUNS/prototypes_448.pt` (Qian's, all 45 images/class, sha256 5abf68ac…)
- val acc (ours / CodaBench): no internal holdout (all 45 images/class) / **0.825 plain** (ID 995914), **0.827 crop 0.75 + flip** (ID 995918; uploaded as `val_team_08_full_384_tta_crop075_flip.zip`, but the file is this run's `tta_crop075_flip/` prediction, not the 384 run's, whose flip score is 0.821, ID 992482), **0.828 crop 0.75 + zoom** (ID 995951, team best). Against 384 (`full_ft_20261016_052544`): +0.8 plain, +0.6 flip, +0.4 zoom.
- trainable % / peak VRAM / GPU-hours: 100% (93,930,241 incl. head) / 17.80 GB / 2.41 (144.0 training minutes, time limit reached)
- what changed vs `van_full_ft_384_fast.yaml`: `img_size` 384 → **448**, `prototypes_448.pt`, batch 128 x 2 → **64 x 4** (same effective 256). ~285 img/s, **5.5 epochs**.
- why: Van's laptop screen at equal compute (10 images/class, crop 0.9): 336 / 384 / 448 / 512 = 63.20 / 64.06 / 64.79 / 64.98%, i.e. +0.73 for 384 → 448 and flat after; CodaBench plain agrees (+0.8).
- latency (A100, forward only, synthetic 448 px, bf16): 7.16 ms at batch 1.
- next: **Task 1 final** (crop 0.75 + zoom, 0.828). 512 is not worth a run (flat on the screen). As at 384, zoom and flip add less at higher resolution (+0.3 / +0.2 over plain at 448 vs +0.7 / +0.4 at 384).

### 2026-10-17 — Van — AdaptFormer + biases/LayerNorms at 384 px (`configs/van_adaptformer_bias_384_fast.yaml`)

- run_dir: `CS604_RUNS/van_adaptformer_20261016_142837` (A100 40 GB, `notebooks/van_colab_launcher.ipynb`, branch `van-dev` 2bd8e78); prototypes `CS604_RUNS/prototypes_384.pt`
- val acc (ours / CodaBench): no internal holdout (all 45 images/class) / **0.794 plain** (ID 994764), **0.807 crop 0.75 + zoom** (ID 994759, Task 2 best; 336: 0.802, ID 980309). Crop 0.75 + flip (`tta_crop075_flip/`) not submitted; its 336 counterpart is 0.794 (ID 980317). The 336 run with biases/norms has no plain submission (the older 336 final without them: 0.779).
- trainable % / peak VRAM / GPU-hours: 3,672,588 backbone params ≈ **4.27%** of ViT-B (head excluded; 11.67% with it) / 19.87 GB / 2.40 (144.0 training minutes, time limit reached)
- what changed vs `van_adaptformer_bias_fast.yaml`: `img_size` 336 → **384** and 384 prototypes; recipe and speed options unchanged. ~447 img/s, **8.6 epochs**.
- why: on Van's equal-compute laptop screen (10 images/class, fixed 5/class holdout) 384 beat 336 by +1.55 at the default crop for AdaptFormer (+0.86 for full FT). On CodaBench the zoom gain is +0.5, as for full FT (+0.5 with zoom, +1.0 plain): zoom and resolution overlap.
- latency (A100, forward only, synthetic 384 px, bf16): 9.86 ms at batch 1, 0.95 ms/image at batch 64.
- training loss rises over epochs 2-6 and then falls, as in the 336 run: the CosFace margin times the learned cosine scale, not a fault.
- side note (laptop screen, full FT, equal compute, crop 0.9): 336 / 384 / 448 = 63.20 / 64.06 / 64.79%, so 448 is still +0.73 over 384 with only 4.1 epochs; a 512 point is not run yet. Laptop runs with a per-epoch holdout (224, 12 epochs) show no accuracy overfitting (holdout accuracy rises every epoch; holdout loss bottoms at epoch 10), and the finals get ~8 epochs.
- next: **Task 2 final candidate** (crop 0.75 + zoom, 0.807).

### 2026-10-16 — Van — full FT at 384 px (`configs/van_full_ft_384_fast.yaml`)

- run_dir: `CS604_RUNS/full_ft_20261016_052544` (A100 40 GB, `notebooks/van_colab_launcher.ipynb`, branch `van-dev` 7e05162); prototypes `CS604_RUNS/prototypes_384.pt` (built on Van's laptop with `scripts/build_prototypes.py`, all 45 images/class)
- val acc (ours / CodaBench): no internal holdout (all 45 images/class) / **0.817 plain** (ID 992465), **0.821** crop 0.75 + flip (ID 992482), **0.824 crop 0.75 + zoom** (ID 992449). Against the same recipe at 336 (Anastasia's `full_ft_20261006_122039`): 0.807 plain, 0.813 crop 0.7 + flip, 0.819 crop 0.75 + zoom, so **+1.0 plain**, +0.8 flip, +0.5 zoom.
- trainable % / peak VRAM / GPU-hours: 100% (93,770,497 incl. head) / 25.34 GB / 2.41 (144.0 training minutes, time limit reached)
- what changed vs `van_full_ft_336_fast.yaml`: `img_size` 336 → **384** and 384 prototypes; everything else identical (lr 4e-4 with layer decay 0.75, head lr 4e-3, label smoothing 0.1, drop_path 0.1, batch 128 x 2, fused AdamW + torch.compile, 12 workers). ~412 img/s, **7.9 epochs**.
- why: on Van's laptop screen at equal compute (10 images/class, same time budget per arm, fixed 5/class holdout) 384 beat 336 by +0.86 at the default crop with 30% fewer epochs; training at 224 then switching to 384 for the last quarter lost 4.3 points. CodaBench (+1.0 plain) agrees with the screen.
- latency (A100, forward only, synthetic 384 px, bf16): 7.02 ms at batch 1, 0.83 ms/image at batch 64.
- side note: holdout error analysis on the 336 laptop checkpoint: top-5 is 93% vs 79% top-1, and 46% of errors predict a class within ±10 class ids of the truth (0.14% by chance), so ids seem to follow the taxonomy and the remaining errors are mostly look-alike neighbours. Label smoothing aimed at those neighbours had no effect on the screen.
- next: **Task 1 final candidate** (best: crop 0.75 + zoom, 0.824). Tighter crops help less at 384 than at 336 (+0.4 flip and +0.7 zoom over plain, vs +0.6 and +1.2).

### 2026-10-15 — Van — speed check for 336 px training (`scripts/van_speed_check.py`)

- run_dir: no training; timing only, Colab A100 40 GB (12 CPUs, torch 2.11), launcher section 4a with `SPEED_CHECK = True`; results `CS604_RUNS/van_speed_check_20261015_051624.json`
- question: every 336 px run trains at about 440-475 img/s whatever the method (full FT ~442, AdaptFormer ~462, adapter ~473), although AdaptFormer is 1.3x faster than full FT on a laptop. What sets the pace?
- results (img/s; full FT 336 / AdaptFormer + biases/norms, batch 128 x 2):

| Measurement | Full FT | AdaptFormer |
|---|---|---|
| Data loader alone, 8 / 10 / 12 workers (colour jitter on) | 507 / 767 / 865 | 568 / 838 / 938 |
| Data loader alone, 8 / 10 / 12 workers, no colour jitter | 856 / 1,143 / 1,242 | 1,078 / 1,292 / 1,461 |
| GPU step alone (synthetic images): plain / fused AdamW / fused + `torch.compile` | 475 / 477 / **600** | 500 / 500 / **670** |
| GPU step alone at batch 256 x 1 (plain / fused + compile) | 486 / 610, 38 GB peak | 517 / 684, 31 GB peak |
| Both together as configured (8 workers, plain) | **447** (the real run: ~442) | **467** (the real run: ~462) |
| Both together, 12 workers + fused + compile | **554 (+24%)** | **622 (+33%)** |

- conclusion: with 8 workers the loader and the GPU step are both near 450-500, so fixing either alone gains little; compile (+26-34% on the GPU step) plus 12 workers removes both. Fused AdamW is +0.4%; colour jitter need not go once 10-12 workers load. 256 x 1 adds only 2% and peaks at 38 GB, so 128 x 2 stays.
- effect: in 144 minutes full FT goes from ~8.5 to ~10.5 epochs, AdaptFormer from ~8.9 to ~11.9. Configs: `configs/van_full_ft_336_fast.yaml` and `configs/van_adaptformer_bias_fast.yaml` (`fused_adamw`, `torch_compile`, `num_workers: 12`; training math unchanged).
- next: final runs of both configs; compare `epoch_time_s` with ~1,018 s (full FT) and ~972 s (AdaptFormer) per epoch.

### 2026-10-12 — Van — AdaptFormer + trainable biases and LayerNorms (`configs/van_adaptformer_bias_final.yaml`)

- run_dir: `CS604_RUNS/van_adaptformer_20261012_060248` (A100 40 GB, `notebooks/van_colab_launcher.ipynb`, branch `van-dev`)
- val acc (ours / CodaBench): internal holdout (laptop, 3 epochs) **76.76%** vs 76.26% without the biases/norms, **78.48%** vs 77.95% with crop 0.75 + zoom / **CodaBench 0.802** with crop 0.75 + zoom (ID 980309), +0.5 over the AdaptFormer final without them (0.797, ID 972126), matching the holdout's +0.53; **0.794** with crop 0.75 + flip (ID 980317; 0.790 without them), so zoom beats flip by +0.8 as on the holdout (+0.79)
- trainable % / peak VRAM / GPU-hours: 3,672,588 backbone params = **4.27%** of ViT-B (head excluded; 11.68% with it) / 15.61 GB / 2.40 (144 training minutes, time limit reached)
- what changed vs `van_adaptformer_final.yaml`: `adapter_train_bias: true` and `adapter_train_norm: true`, so the pretrained backbone's biases (BitFit) and LayerNorm weights also train, at the adapter lr, with no weight decay (122,112 extra parameters). They cost about 10% throughput on the laptop; on the A100 the run did ~462 img/s, 8.9 epochs at batch 128 x 2.
- training loss rose over epochs 2-6 (5.34 -> 5.65) and then fell to 4.57: the CosFace margin times the learned cosine scale, not a fault.
- latency (A100, forward only, synthetic 336 px, bf16): 10.28 ms at batch 1, 0.73 ms/image at batch 64.
- next: use this as the Task 2 submission recipe.

### 2026-10-09 — Van — zoomed second view at prediction (`src.predict --zoom`)

- run_dir: no training; prediction-only option for any trained checkpoint (`src/van_zoom.py`, launcher section 6c)
- val acc (ours / CodaBench): internal holdout (5 training images/class held out), eval crop 0.75, 2 forward passes each: plain + zoom vs plain + flip **77.95 vs 77.10%** (AdaptFormer, 3 laptop epochs) and **79.59 vs 78.97%** (full FT at 336, 3 laptop epochs); 4 passes (with both flips) 78.08 / 79.63% / **CodaBench 0.797** for the AdaptFormer final (ID 972126; was 0.790 with crop 0.75 + flip) and **0.819** for Anastasia's full FT at 336 `full_ft_20261006_122039` (ID 972132; was 0.813 with crop 0.7 + flip): +0.7 and +0.6, as the holdout predicted. Same checkpoints, no retraining, predicted with `--crop_pct 0.75 --zoom`.
- trainable % / peak VRAM / GPU-hours: unchanged (inference only); 2 passes per image, like crop + flip
- what it is: after the usual pass, the patch tokens most similar to the predicted class's head weight mark where the model found its evidence; the smallest square holding 70% of that map is cut from a 2x-resolution copy of the image, enlarged to the input size and classified again; the softmaxes are averaged. A centred square of the same size scored 77.67 / 79.43%: most of the gain is the close-up itself, the rest is zooming to the right place (largest for small subjects: +2.5 vs +0.9 on the 691 images with the tightest boxes, full FT).
- next: use `--zoom` for the Phase 2 test predictions; the launcher's section 6c does it for new runs.

### 2026-10-09 — Van — AdaptFormer (Task 2) tuned on a laptop (`configs/van_adaptformer_final.yaml`)

- run_dir: laptop screens in Van's own pipeline (RTX 5080 laptop, 336 px, 1 epoch per arm unless noted); final run `af_final_45` on Colab A100 40 GB from Van's personal pipeline (same recipe as `configs/van_adaptformer_final.yaml`, which is the team-trainer port: identical logits on a trained checkpoint)
- val acc (ours / CodaBench): internal holdout (5 training images/class held out, trained on the other 40) **76.26%** after 3 epochs, **77.13%** with crop 0.75 + flip / **CodaBench 0.779** plain (ID 971289), **0.790** with crop 0.75 + flip (ID 971291); all 45 images/class, 2.0 x 48 GB-hours (8,756 s), ~10 epochs
- trainable % / peak VRAM / GPU-hours: 3,550,476 backbone params = **4.13%** of ViT-B (head excluded) / 7.6 GB at batch 64 on the laptop / 3 epochs = 1.9 laptop hours (0.63 of the 48 GB-hour budget)
- what it is: `method: van_adaptformer` (new, opt-in; the team's `adaptformer.py` stub is untouched): a parallel bottleneck (768 -> 192 -> 768, ReLU) beside every block's MLP, output zero at the start, trained at lr 1e-3; prototype cosine head at 4e-3; label smoothing 0.1; clip 1.0; batch 256; 5% warmup + cosine fitted to 144 min. No layer decay or drop_path (full-FT only).
- findings, one change at a time against a plain AdaptFormer (71.49% at 1 epoch): adapter lr 3e-4 / 1e-3 / 3e-3 / 1e-2 = 70.56 / 71.49 / 71.49 / 70.45; **loss design**: CosFace margin 0.1 (`cosine_margin`) **+0.60** (margin 0.2 -0.54; no label smoothing -0.32); **initialization/design**: adapter reads norm2(x) instead of the residual stream +0.24, learned per-block scale +0.15, PCA init of the down projection (`adapter_init: pca`) +0.08; width 64 (1.38%) -0.56. All four positive changes together, 3 epochs: **76.26% vs 75.27%** for the plain control (+0.99, ahead at every epoch).
- learned scales rise with depth (blocks 1-11: 0.03-0.09, block 12: 0.15): the model wants more adaptation in deep blocks.
- latency (forward only, synthetic 336 px, bf16, 30 warm-ups + 200 iterations, **RTX 5080 laptop**, not A100): 5.60 ms at batch 1 (10.58 ms with flip), 1.83 ms/image at batch 64; the same ViT-B without adapters 4.04 / 1.70 ms. Adapters cannot be merged like LoRA: +39% at batch 1, +7% at batch 64.
- next: an A100 latency to compare with the team's numbers; later runs of this recipe go through `notebooks/van_colab_launcher.ipynb` (METHOD `van_adaptformer_final`).

### 2026-10-06 — Van — full FT recipe tuned on a laptop (`configs/van_full_ft_proto_llrd.yaml`)

- run_dir: laptop screens in Van's own pipeline (RTX 5080 laptop); the Colab run of this config (A100 40 GB, `notebooks/van_colab_launcher.ipynb`, branch `van-dev` at f5d654a) is under `CS604_RUNS`, folder name TBC
- val acc (ours / CodaBench): internal holdout (5 training images/class held out, trained on the other 40) **72.98%** after 3 epochs, **73.97%** after 6 / **CodaBench 0.757** (ID 963899, 2026-10-06; all 45 images/class, 144 training minutes, plain center crop 0.9, no test-time augmentation) / **CodaBench 0.773** (ID 964799, same `last.pt` re-predicted with `--crop_pct 0.7 --tta_flip`, no retraining)
- trainable % / peak VRAM / GPU-hours: 100% / 5.8 GB at batch 64 / 6 epochs = 2.1 laptop hours (about 0.7 of the 48 GB-hour budget)
- what changed vs `full_ft_proto.yaml`: prototype-initialized **cosine** head; backbone lr 4e-4 with **layer-wise lr decay 0.75** (block i gets lr x 0.75^(12 - i)); head lr 4e-3; 5% warmup then cosine to zero **fitted to the time limit**; label smoothing 0.1; drop_path 0.1; gradient clipping 1.0; batch 256.
- findings (one change at a time, 3 epochs): without layer decay 63.1% (vs 69.8%); head lr 10x the backbone +1.5 points; prototype start beats first training a linear probe (LP-FT) by 1.6–2.0 points; crop scale, colour jitter and head lr 30x no difference; seed-to-seed spread 0.33 points.
- trainer additions (all off by default, so existing configs run exactly as before): `layer_decay`, `drop_path_rate`, `clip_grad_norm`, `schedule_fit_time` (with `max_train_minutes`, the cosine reaches zero at the time limit instead of being cut off mid-decay).
- prediction: center crop 0.7 + flip averaging added +1.5 to +1.9 points on the internal holdout and +1.6 on CodaBench. A 12-epoch laptop run reached 74.49% on the holdout (+0.5 over 6 epochs), so training length is nearly saturated. Details: `report/van_full_ft_findings.md`.
- next: fill in epochs reached, peak VRAM, GPU-hours and latency from the run folder; try more regularization in long runs; Task 2 on the same recipe.

## Carry-overs (from other branches; not rerun in this trainer yet)

### Qian — full FT CodaBench (`qq_branch`)

- Official backbone `vit_base_patch16_224.augreg2_in21k_ft_in1k`, batch 32, bf16, AdamW, no scheduler.
- **Two LRs:** backbone `1e-5`, head `1e-3` (now `configs/full_ft_proto.yaml` + `head_lr` in the trainer).
- 120 min → **CodaBench 0.610**. 144 min (~2.4 GPU-hours on A100) → **0.613**.
- Same + class-mean head init → **0.621**. Latency ~**7.15 ms/image** (batch 1, A100).
- Peak PyTorch alloc ~**3.35 GiB**. Submission name she used: `val_team_08.txt`.

### Van — prototype backbone screen (`van` branch)

Same holdout (5 images/class, 50k), no training, cosine / class means:

| Backbone | Internal top-1 |
|---|---|
| `orig_in21k` | 41.8% |
| `augreg2_in21k_ft_in1k` (course default) | 43.9% |
| `augreg_in21k` (21k only) | **50.0%** |
| DINOv2 ViT-B/14 at 224 | 53.8% (ask prof before counting this) |

Prototype + LoRA r8 on the default backbone: ~**51–53%** internal (not CodaBench).

Prof (Van’s notes, 2026-10-03): head is always trained and **not** counted in the 5% PEFT cap; comparable-size ViTs allowed.

### 2026-10-02 — Van — prototype head (ported here)

- class-mean features, no training → ~43.9% (PhD zero-shot line)
- random head → ~0.28%
- in this trainer: `scripts/build_prototypes.py` + `head_init: prototype`

### 2026-10-02 — Anastasia — Backbone screen (frozen linear probe, screening subset)
 
Frozen features + linear probe, internal holdout accuracy:
 
| Backbone | Linear probe top-1 | Throughput (img/s) |
|---|---|---|
| `augreg2_in21k_ft_in1k` | **59.0%** | 1,071 |
| DINOv2 ViT-B/14 | 58.6% | 734 |
| DeiT III | 52.6% | – |
| CLIP ViT-B | 41.7% | – |
 
- augreg2 and DINOv2 tied within noise (~1 point); DeiT III and CLIP dropped.
- Caveat: the best probe LR was 3e-3 for every model, the top of the sweep, so all probe numbers may be slightly low. Add 1e-2 and 3e-2 next time.

#### Resolution ablation (full FT, 2,000-class subset, equal time budget) — date TBC
 
- Arms: A = 224, B = 336, C = 224 then 336. Time budget scaled to the subset so each arm gets roughly the epoch count of a full run (augreg2 at 336 completed 9 epochs).
- **336 beat 224 for both backbones** despite fewer epochs; augreg2 gained **+3.7 points** at 336.
- DINOv2 beat augreg2 at every resolution once fine-tuned.

### 2026-10-03 — Anastasia — frozen DINOv2 linear probe @ 336
 
- run_dir: notebook run (Drive), TBC
- val acc (ours / CodaBench): 0.737 holdout / **0.742**
- trainable % / peak VRAM / GPU-hours: 0% backbone (head only) / TBC / TBC
- what changed: DINOv2 backbone instead of augreg2
- next: **not valid for submission** (prof requires augreg2). Holdout closely predicted CodaBench (0.737 vs 0.742), which supports using the 5/class holdout for model selection.

### 2026-10-03 — Anastasia — frozen augreg2 linear probe @ 224
 
- run_dir: notebook run (Drive), TBC
- val acc (ours / CodaBench): TBC / **0.625** (`val_team_8_probe_augreg2_224.zip`)
- trainable % / peak VRAM / GPU-hours: 0% backbone (head only) / TBC / TBC
- what changed: official backbone, frozen
- next: baseline to beat for all PEFT runs

### 2026-10-04 — Anastasia — frozen augreg2 linear probe @ 336
 
- run_dir: notebook run (Drive), TBC
- val acc (ours / CodaBench): **0.611** holdout top-1, top-5 0.818 / **0.620**
- trainable % / peak VRAM / GPU-hours: 0% backbone (head only, 7.69M params, excluded from the cap) / TBC / feature extraction 11.3 min + probe 2.4 min
- what changed: resolution 224 → 336; best LR 1e-3 from the sweep below
- next: no gain for frozen features (within noise of 0.625). Likely because augreg2 was pretrained at 224, so frozen weights can't use the extra detail. Contrast with the +3.7 points from 336 when fine-tuned: **higher resolution seems to pay off only when the model is adapted.**

### 2026-10-04 — Anastasia — frozen augreg2 linear probe @ 336, LR sweep (same run as the 0.620 submission above)
 
- run_dir: notebook run (Drive), TBC
- val acc (ours / CodaBench): best **0.611** holdout top-1 at LR 1e-3 (top-5 0.818) / **0.620** (the LR 1e-3 head is the one submitted)
- trainable % / peak VRAM / GPU-hours: 0% backbone (head only) / TBC / TBC
- what changed: LR sweep 1e-3, 3e-3, 1e-2, 3e-2; 30 epochs each, one-cycle schedule

| LR | Best holdout top-1 | Final holdout top-1 | Final holdout loss |
|---|---|---|---|
| 1e-3 | **0.611** | 0.611 | 3.13 |
| 3e-3 | 0.608 | 0.608 | 3.25 |
| 1e-2 | 0.587 | 0.587 | 3.59 |
| 3e-2 | 0.516 (epoch 2) | 0.241 | 10.13 (diverged) |
 
- results: 1e-3 and 3e-3 are tied within noise; higher LRs are worse, and 3e-2 collapses. This closes the earlier worry that probes were capped by the top of the LR sweep: the frozen augreg2 probe ceiling is about 0.61 holdout, close to CodaBench 0.620 (holdout ran about 1 point below CodaBench here, and about 0.5 below for the DINOv2 probe). At 1e-3, holdout accuracy plateaus from about epoch 19 and holdout loss bottoms out at epoch 16 (3.11), so 30 epochs is more than needed.
- next: frozen baseline for all comparisons on the same holdout and resolution. LoRA r16 @ 336 (0.719) is about +11 points above it.

### 2026-10-03 — Anastasia — LoRA r16 (qkv, proj, fc1, fc2, all 12 blocks) @ 224, LR 1e-3
 
- run_dir: notebook run (Drive), TBC
- val acc (ours / CodaBench): stopped early, not submitted
- trainable % / peak VRAM / GPU-hours: ~2.4M LoRA params, roughly 2.5–2.8% depending on the denominator used (head excluded) / TBC / partial
- what changed: LoRA on top of a head pre-trained on the 400k fit images only (the earlier probe head saw the holdout, so it wasn't reused)
- next: training loss was rising, which suggests LR 1e-3 was too high. Stopped and rerun at 336 with a lower LR. Throughput ~830 img/s at 224, similar to full FT, since LoRA still runs the full forward and backward pass.

### 2026-10-04 — Anastasia — LoRA r16 (same targets) @ 336, LR 3e-4
 
- run_dir: notebook run (Drive), TBC. History: `lora_augreg2_336_r16_history.csv`
- val acc (ours / CodaBench): **0.719** holdout top-1 (best epoch 6: 0.7194; final 0.7192), top-5 0.885 / CodaBench TBC
- trainable % / peak VRAM / GPU-hours: as above / TBC / ~2.4 (145.2 min on A100, including per-epoch holdout checks)
- what changed: resolution 224 → 336; LoRA LR 1e-3 → 3e-4 (head LR 1e-4, weight decay 1e-4, AdamW, 5% warmup then cosine to zero over the time budget)
- results: ~371 img/s, ~7.5 epochs. Holdout top-1 0.678 after epoch 1, then +1.2, +1.0, +0.9, +0.4, +0.5, +0.2, 0.0 points. Holdout loss flat from epoch 6 (~2.66) vs train loss ~1.50. Gains flatten as the cosine LR approaches zero, so the plateau is at least partly the schedule.
- next: submit to CodaBench (holdout tracked CodaBench closely before, so expect roughly 0.72). Compare with probes (0.620 / 0.625) and Qian's full FT (0.621). Candidate follow-ups: LR 5e-4, FixRes (224 then 336) for more epochs in the same budget, and smaller ranks / placements from Cluster 4.


### 2026-10-05 — Qian — Prototype-initialized full fine-tuning (shared trainer)

- branch / config: `qq_branch_v2` / `configs/full_ft_proto.yaml`; launcher: `notebooks/colab_launcher_qq.ipynb`
- run_dir: `full_ft_20261005_074553_id962394` under Drive `CS604_RUNS`; renamed after submission. The latency JSON retains the original checkpoint path; it refers to this same run’s `last.pt`.
- CodaBench top-1 accuracy: **0.663 (66.3%)**; submission ID: **962394**
- backbone / head: `vit_base_patch16_224.augreg2_in21k_ft_in1k` at 224px / linear classifier initialized from normalized training-class prototypes
- settings: AdamW; backbone LR `1e-5`; head LR `1e-3`; cosine decay; no warmup; weight decay 0.05; batch 32; gradient accumulation 1; BF16; seed 42
- trainable parameters: **93,488,656 / 93,488,656 (100%)**, including the classifier head.
- budget: 144 training minutes or 11 epochs, whichever comes first; excludes setup, prototype loading, validation and checkpoint saving
- actual training: **8,640.15 seconds (144 minutes), 2.40004 training GPU-hours** on one GPU; stopped at the time limit (`time_limit_reached=True`). Total recorded wall time: 8,648.13 seconds (2.40226 hours), including setup and other operations.
- peak allocated VRAM: **3.349857 GiB** (saved as `peak_vram_gb`).
- inference latency: **6.522547 ms/image at batch size 1**; **0.277780 ms/image at batch size 64** (17.777900 ms/batch). BF16, 224px, 30 warm-ups and 200 timed iterations; forward pass on synthetic inputs, excluding loading, preprocessing and transfers. PyTorch `2.11.0+cu130`; results saved in `inference_latency.json`.
- hardware: **NVIDIA A100-SXM4-40GB**, reported capacity 39.493530 GiB; one GPU; **High-RAM off** (confirmed by Qian).
- loss curves: saved in the run’s `figures/` folder (confirmed by Qian); raw training losses are retained in the CSV logs.
- result: **+4.2 percentage points** over the previous prototype full-FT result (0.621). This run uses the shared trainer's cosine schedule; multiple implementation/runtime differences prevent attributing the gain to one change.

### 2026-10-06 — Anastasia — full FT @ 336, backbone-LR pilot (`configs/full_ft_336.yaml`)

- run_dir: `CS604_RUNS/pilot_full_ft_336_20261006_013706`
- val acc (ours / CodaBench): pilot holdout only (1,000-class subset, 5/class held out, subset seed 0), not comparable to full-scale or CodaBench numbers / not submitted
- trainable % / peak VRAM / GPU-hours: 100% / 6.27 GB at batch 32 / ~0.6 (3 runs × ~11.6 min)
- what changed: backbone lr sweep 1e-5, 3e-5, 1e-4; 3 epochs each; head lr 1e-3; no layer decay; batch 32 × grad accum 2

| Backbone lr | Holdout top-1 | Holdout loss |
|---|---|---|
| 1e-5 | 0.8772 | **0.481** |
| 3e-5 | **0.8798** | 0.485 |
| 1e-4 | 0.8542 | 0.587 |

- results: 1e-5 and 3e-5 tied (0.26 points, inside the 0.3 tie tolerance), so the pilot chose 1e-5; 1e-4 was 2.6 points worse. All runs still improving at epoch 3, no early dip. Without layer decay, higher lrs hurt, consistent with Van's −6.7 points for 4e-4 without layer decay.
- note: this pilot ran with `num_workers: 2` and was data-loading bound (the head-init test below ran the same data 2× faster with all CPU workers).
- next: superseded by Van's layer-decay recipe (4e-4 top lr is safe only with layer decay). Use `configs/full_ft_336_llrd.yaml` for the 336 vs 224 comparison.

### 2026-10-06 — Anastasia — head initialisation test, full FT @ 336 (pilot scale)

- run_dir: launcher cell "Head initialisation test" (`colab_launcher.ipynb`, `ana-branch`), same pilot subset and holdout as the LR pilot above
- val acc (ours / CodaBench): pilot holdout only (1,000 classes, 5/class) / not submitted
- trainable % / peak VRAM / GPU-hours: 100% / TBC / ~0.4 (4 arms × 5.7 min)
- what changed: only the starting head. Backbone lr 1e-5 (from the pilot), head lr 1e-3, 3 epochs, no layer decay. Prototype and probe heads built from the pilot's fit images only, so the holdout stays clean. One seed per arm.

| Starting head | Holdout top-1 before training | Final holdout top-1 | Final holdout loss |
|---|---|---|---|
| A: random | – | **0.8782** | 0.490 |
| B: class prototypes | 0.669 | 0.8776 | **0.484** |
| C: linear probe (LP-FT) | 0.799 | 0.8558 | 0.606 |
| D: linear probe, head lr 1e-4 | 0.799 | 0.8554 | 0.623 |

- results: **LP-FT is about 2.2 points worse** than random or prototype starts, despite starting 13 points higher. This independently confirms Van's finding (prototype beats LP-FT by 1.6–2.0 points at full scale, 224, with layer decay) in a different setting (336, no layer decay, lr 1e-5). Lowering the head lr (D) does not help, so the head is not being disrupted; more likely the confident probe head gives small gradients and the backbone adapts less (Van's explanation). Random and prototype tie at this scale (0.06 points, within seed noise), likely because a 100× head lr lets a random head catch up within 3 epochs on 1,000 classes; at 10,000 classes Qian saw prototype ahead (0.613 → 0.621 on CodaBench).
- speed: 5.7 min per arm vs ~11.5 min for the same data in the LR pilot; the only difference is `num_workers` (all CPUs vs 2), so full FT at 336 is data-loading bound with 2 workers.
- next: keep the prototype head for full FT; drop LP-FT (`probe_head.pt` stays as the frozen baseline only). Run `configs/full_ft_336_llrd.yaml` (Van's recipe at 336, 8 workers) and compare with Van's 0.757 at 224.

### 2026-10-06 — Anastasia — full FT @ 336 with Van's recipe (`configs/full_ft_336_llrd.yaml`)

- run_dir: `CS604_RUNS/full_ft_20261006_122039` (launcher `notebooks/colab_launcher_ana_ft336.ipynb`, `ana-branch`)
- val acc (ours / CodaBench): no internal holdout (trained on all 45 images/class) / **0.807** plain center crop 0.9; **0.813** with crop 0.7 + flip TTA
- trainable % / peak VRAM / GPU-hours: 100% / 20.05 GB / 2.40 (144.0 training minutes, time limit reached; A100 40 GB)
- what changed vs Van's `van_full_ft_proto_llrd.yaml` (0.757): img_size 224 → **336**; batch 256 × 1 → 128 × 2 (same effective batch 256, half the activation memory); prototypes rebuilt at 336 (`prototypes_336.pt`). Everything else identical: prototype cosine head, lr 4e-4 with layer decay 0.75, head lr 4e-3, label smoothing 0.1, drop_path 0.1, clip 1.0, 5% warmup + cosine fitted to 144 min.
- results: **+5.0 points over 224 at the same time budget**, despite ~8.5 epochs (~17 min/epoch, ~440 img/s) vs Van's ~16 at 224. Far beyond seed noise (~0.3). Train loss still falling at the end (1.90 → 1.87 over the last epoch). Consistent with the earlier resolution ablation (+3.7 at 336 when fine-tuned) and with frozen probes showing no gain at 336: higher resolution pays off when the backbone adapts. Crop 0.7 + flip added +0.6 (smaller than Van's +1.5–1.9 at 224 on the internal holdout; the best crop at 336 may differ from 0.7).
- cost: latency **7.94 ms/image** at batch 1 (0.64 ms/image at batch 64), forward only, bf16; only ~20% above the team's 224 full-FT batch-1 numbers (6.5–7.2 ms), since batch 1 leaves the GPU underused. TTA needs two forward passes (~2× latency, not measured). Prototype build at 336 is outside the training budget, as for Van's run.
- next: team best so far. Candidate follow-ups: other eval crops at 336 on the same checkpoint (no retraining); Task 2 methods at 336 with the same recipe pieces.


### 2026-10–08 — Qian — selected-attention internal-validation ablation

- source branch / launcher: `qq_selected_layer_validation` / `notebooks/colab_qq_selected_layer_validation.ipynb`; saved notebook commit `1be03d8`.
- purpose: select which attention parameters to adapt while training <5% of backbone parameters, excluding the classifier.
- data: fixed internal split of the labelled training set, 40 fitting + 5 validation images/class across all 10,000 classes (400,000 fitting / 50,000 holdout); split seed 42. This holdout is distinct from official CodaBench validation. Split manifests and raw run artifacts are retained under Drive `CS604_RUNS/qq_selected_layer_<split-id>/`.
- shared recipe: `configs/linear_probe.yaml` (inheriting `base.yaml`); augreg2 ViT-B/16 at 224; random linear classifier initialization, no prototypes; AdamW LR 1e-3 for all trainable parameters, weight decay 0, batch 64, accumulation 1, 1-epoch warmup followed by cosine decay, BF16, training seed 42, 2 data-loader workers. Each arm completed 5 epochs; no training-minute cutoff. GPU model/session equivalence is not established by the supplied summary.
- selective scopes: QKV includes weights and biases; complete attention includes QKV and the output projection. MLPs and all LayerNorms remain frozen in the selective arms. The 7,690,000-parameter classifier is trained in every arm and excluded from the backbone cap; backbone denominator is 85,798,656.

| Arm | Run folder | Best / final holdout top-1 | Best epoch | Trainable backbone % | Peak VRAM (GiB) | Training minutes | Wall GPU-hours |
|---|---|---:|---:|---:|---:|---:|---:|
| Classifier only | `linear_probe_20261007_083325` | 0.62700 | 5 | 0.000000 | 0.787727 | 109.97 | 2.052532 |
| QKV in last block | `last_blocks_20261007_120432` | 0.64914 | 5 | 2.065039 | 0.888645 | 115.26 | 2.144944 |
| QKV in last two blocks | `last_blocks_20261008_013844` | 0.64056 | 5 | 4.130079 | 1.151424 | 112.62 | 2.094281 |
| Complete attention in last block | `last_blocks_20261008_034512` | **0.65410** | 5 | 2.753386 | 0.894388 | 109.94 | 2.048139 |

- results: all selective arms outperform classifier-only training. QKV in one block adds 2.214 percentage points; complete attention adds 2.710 points. Training the output projection alongside QKV adds 0.496 points over QKV-only in the same block, with nearly identical measured VRAM. QKV in two blocks is 0.858 points worse than one-block QKV and uses more memory under this recipe.
- selected configuration: **complete attention in the last block** for the next all-data experiment, based on the highest holdout accuracy while using 2.753386% of backbone parameters. This is a selection under a shared five-epoch recipe, not a claim of each method's optimized performance.
- loss-curve observation: for the selected arm, screenshot-derived validation loss falls from approximately 3.07 at epoch 1 to 1.60 at epoch 4 and 1.59 at epoch 5, while training loss falls from about 4.87 to 1.33. Validation loss nearly plateaus; there is no visible validation-loss increase. These approximate readings do not replace the raw CSV.
- limitations: best accuracy occurs at epoch 5 in all four arms; convergence is not established. One seed per arm does not establish the robustness of the 0.496-point lead. Cosine LR approaches zero at the five-epoch endpoint, which may contribute to flattening. Timing differences across sessions do not establish a speed advantage. Wall GPU-hours include overhead and are distinct from training-only duration. No CodaBench score or inference-latency result is recorded for these development runs.
- next: retrain the selected scope from the pretrained model on all 450,000 labelled training images using a predeclared final-run schedule and resource limit, then evaluate official predictions. The existing 66.3% full-FT run uses a different recipe (prototype initialization, learning rates, batch size, regularization, and schedule), so it is an additional reference rather than a fully controlled adaptation comparison.


### 2026-10-08 — Qian — last-block attention, prototype head (final all-data run)

- config / run: `configs/lastblock_attention.yaml` on `qq_branch_v2`; launcher `notebooks/colab_launcher_qq.ipynb`; `CS604_RUNS/last_blocks_20261008_123344/last.pt`.
- CodaBench: **0.668 (66.8%)**, submission **969549**, confirmed to use this checkpoint; standard launcher evaluation, no TTA. No internal holdout in this run.
- design: train final-block QKV and attention output projection plus the prototype-initialized linear classifier; remaining backbone frozen. **2.753386% backbone trainable**, classifier excluded; 10,052,368 trainable parameters including classifier.
- recipe: augreg2 ViT-B/16 at 224; batch 64, accumulation 1; AdamW LR 1e-3, weight decay 0; BF16, seed 42; 5% time-based warmup + cosine decay. Reused all-training-data prototypes.
- cost: **144.0 training minutes**, no epoch ceiling; **5 complete epochs + 96.8% of epoch 6** before the time limit. A100-SXM4-40GB; peak allocated VRAM **0.8934 GiB**; wall GPU-hours **2.40076**.
- latency: **7.693 ms/image** at batch 1; **0.289 ms/image** at batch 64, synthetic forward-only BF16 (30 warmups, 200 iterations).
- loss: epoch-average training loss 3.165 → 1.325 over complete epochs 1–5; partial epoch 6: 1.157. Raw logs retained; plots saved in `figures/`.
- comparison: **+0.5 percentage points** over Qian's 66.3% full-FT reference, with fewer trainable backbone parameters and lower measured peak VRAM. Training recipes differ, so this is not an isolated adaptation-method comparison. Scope selection used random-head internal validation; this final run used prototypes.
