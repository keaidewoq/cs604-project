# Full fine-tuning (Task 1): what Van tested and what worked

Author: Van. Written 2026-10-06. Status: final Colab run done, **CodaBench 0.773**
with crop 0.7 + flip prediction (ID 964799); 0.757 with plain prediction (ID
963899), same checkpoint. The 12-epoch laptop check is done (section 9).

This document is written so that a teammate, or a teammate's coding agent, can
read it alone and understand: what the recipe is, why each setting was chosen,
how strong the evidence is, and how to reuse it in this repo. All numbers come
from logged runs; nothing is estimated unless it says so.

## 1. Summary

- **Result.** `configs/full_ft_224px_proto-cosine_llrd0.75_top-lr4e-4_2.4h.yaml`, trained for 144 minutes on
  a Colab A100 40 GB on all 450,000 training images, scored **0.773 on
  CodaBench** when predicted with a center crop of 0.7 plus flip averaging
  (0.757 with the plain 0.9 center crop, same checkpoint). The previous best
  full fine-tune in the team was 0.663.
- **The recipe in one line.** Start the classifier from class prototypes (a
  cosine head), fine-tune everything with AdamW at lr 4e-4 for the top
  transformer block and smaller rates for lower blocks (layer-wise lr decay
  0.75), head lr 4e-3, 5% warmup then cosine decay that ends exactly at the time
  limit, label smoothing 0.1, drop_path 0.1, gradient clipping 1.0, batch 256,
  bf16.
- **What mattered most** (each measured by changing one setting at a time):
  1. A learning-rate schedule that decays to zero (warmup + cosine): one epoch
     with it already beats 11 epochs at a constant rate.
  2. Layer-wise learning-rate decay: removing it cost **6.7 points**.
  3. A higher learning rate than before: 4e-4 (top block) instead of 1e-5.
  4. Starting the head from the class prototypes: **+1.6 to +2.0 points** over
     first training a linear probe (LP-FT).
  5. A head learning rate 10x the backbone's, when starting from a linear
     probe: +1.5 points (from prototypes, 10x and 30x tied).
  6. Training longer: 3 to 6 epochs added +1.0 point; 6 to 12 only +0.5.
- **What did not matter** (within noise): crop-scale augmentation range,
  colour jitter, head lr 30x vs 10x from prototypes.
- **Free extra gain at prediction time:** a tighter center crop (0.7 instead
  of 0.9) plus flip averaging added **+1.5 to +1.9 points** on the internal
  holdout and **+1.6 on CodaBench** (0.757 to 0.773), with no retraining
  (section 8).

## 2. Background: the starting point

The backbone is the team's locked `vit_base_patch16_224.augreg2_in21k_ft_in1k`
(about 86M parameters plus a 10,000-class head, about 7.7M).

Earlier team full fine-tunes (from `report/experiment_index.csv`): 0.610, 0.613
and 0.621 on CodaBench with a constant learning rate (no warmup or decay),
backbone lr 1e-5, head lr 1e-3 (100x), batch 32, no label smoothing or
regularization; then 0.663 with the shared trainer's cosine decay.

Things that looked limiting in those runs, and which this work tested:

- **No learning-rate decay.** The model never settles; batch loss at the end
  still swung between about 0.6 and 3.0. More time (120 to 144 minutes) added
  only +0.3 points.
- **Very low backbone rate with a 100x head rate.** The head moves fast while
  the backbone barely adapts.
- **Small batch and an idle GPU.** Batch 32 used about 3.4 GB of a 40 GB card,
  and training ran at about 570 img/s, limited by data loading.
- **No regularization** (label smoothing, stochastic depth, layer-wise decay)
  with only 45 images per class.
- **When `max_train_minutes` stops training before `epochs` is reached, the
  cosine schedule is cut off part-way**, so the learning rate never reaches
  zero. The new `schedule_fit_time` option fixes this (section 6).

## 3. How the experiments were run

### 3.1 Where

Tuning ran on Van's laptop (RTX 5080 Laptop GPU, 16 GB, full power) in Van's
own training pipeline, which is kept outside this repo because it contains the
internal-holdout code. Only the final run used this repo's trainer on Colab.
Laptop speed for this full fine-tune: about 390 img/s (GPU-bound), so one epoch
over 400,000 images took about 22 minutes.

Config names below such as `full_lpft_a_lr4e-4` are run names in that pipeline,
listed so the results can be traced; they are not files in this repo. The
settings are written out in full in each table.

### 3.2 The internal holdout (how "holdout top-1" is measured)

CodaBench allows few submissions, so tuning used a fixed internal holdout:

- From the 45 labelled training images per class, **5 per class are held out**
  (50,000 images). The other **40 per class (400,000 images) are used for
  training**. The split is fixed (seed 604) and identical in every run.
- "Holdout top-1" in this document is top-1 accuracy on those 50,000 images.
- **Noise level.** One score has a standard error of about 0.2 points
  (50,000 images). Two runs of the same recipe with different seeds differed by
  **0.33 points**. Treat differences under about 0.4 points as ties.
- **Relation to CodaBench.** Both have 5 images per class, but they are
  different images, and the final run trains on 45 instead of 40 images per
  class and for longer, so CodaBench scores are expected to run somewhat
  higher than holdout scores of shorter runs. Holdout numbers are for comparing
  settings with each other, not for predicting CodaBench exactly.
- Every run first evaluates its starting model before any update and checks
  that it reproduces the expected starting accuracy (for example 43.90% for
  the prototype head). All runs below passed this check.

### 3.3 Shared settings unless a table says otherwise

AdamW, weight decay 0.05 (none on norms and biases), label smoothing 0.1,
drop_path 0.1, gradient clipping 1.0, effective batch 256 (64 x 4 gradient
accumulation on the laptop), bf16 autocast, 5% linear warmup then cosine decay
to zero over the run, timm's default training augmentation for this backbone
(random-resized crop with scale 0.08 to 1, horizontal flip, colour jitter 0.4),
evaluation with the backbone's default center crop (resize then crop 0.9),
cosine classifier head kept in float32.

## 4. Experiments and results

### 4.1 Starting points without fine-tuning

| Model | Holdout top-1 |
| --- | --- |
| Prototype head only (class means of frozen features from the 40 training images per class, cosine classifier, temperature 20) | 43.90% |
| Linear probe: the head trained on frozen features for 10 epochs (lr 3e-3, label smoothing 0.1) | 60.37% |

The linear probe was trained on cached features (each image passed through the
frozen backbone once), so a 10-epoch probe takes about 36 seconds. The probe's
learning rate barely mattered (1e-3 to 3e-2 all gave 60.32% to 60.39%). Without
label smoothing it reached 61.06%, but its learned temperature then passed the
head's cap of 100.

### 4.2 Round A: backbone learning rate (1 epoch, LP-FT start)

Full fine-tune for 1 epoch, starting from the linear-probe head (60.37%),
`layer_decay` 0.75, the head sharing the top backbone rate.

| Backbone lr (top block) | Holdout top-1 | Holdout loss |
| --- | --- | --- |
| 5e-5 | 66.24% | 1.658 |
| 1e-4 | 66.82% | 1.621 |
| 2e-4 | 67.22% | 1.586 |
| **4e-4** | **67.47%** | 1.553 |
| 8e-4 | 67.34% | 1.538 |

- The optimum is flat between 4e-4 and 8e-4. At 8e-4 the training loss rose
  early (to about 3.6 near step 350, from about 3.0) before recovering, a sign
  the rate was disturbing the pretrained weights. **4e-4 was kept** because
  longer runs usually prefer a slightly lower rate.
- For comparison, the earlier team setting was 1e-5 with no layer decay.

### 4.3 Round B: head rate and layer decay (3 epochs, LP-FT start)

Backbone lr 4e-4, 3 epochs (the cosine reaches zero at the end of epoch 3).

| Change from the control | Epoch 1 | Epoch 2 | Epoch 3 (final) | Final loss |
| --- | --- | --- | --- | --- |
| Control (head shares 4e-4) | 63.86% | 68.57% | 69.83% | 1.505 |
| **Head lr 4e-3 (10x)** | 62.18% | 69.20% | **71.37%** | 1.400 |
| No layer decay (every layer at 4e-4) | 44.90% | 58.44% | 63.12% | 1.778 |

- **Layer decay is essential at this rate.** Without it, accuracy fell below
  the 60.37% starting point in epoch 1 and ended 6.7 points lower: updating the
  early layers at 4e-4 damages their pretrained features.
- Epoch-1 and epoch-2 scores of a 3-epoch run are lower than a 1-epoch run's
  final score because the learning rate is still high mid-schedule. Compare
  final epochs only.

### 4.4 Round C: augmentation and a faster head (3 epochs)

Each run changes one thing from "head lr 4e-3" above (71.37%).

| Change | Final holdout top-1 | Verdict |
| --- | --- | --- |
| Random-resized-crop scale [0.35, 1.0] instead of [0.08, 1.0] | 71.11% | no effect |
| Colour jitter off (instead of 0.4) | 71.29% | no effect |
| Head lr 1.2e-2 (30x) | 71.97% | +0.6, but see section 4.5 |

Augmentation changes often only pay off in longer runs; at 3 epochs they made
no difference, so timm's defaults were kept.

### 4.5 Head initialization: prototypes beat LP-FT (3 epochs)

Same recipe (backbone lr 4e-4, layer decay 0.75, head lr 4e-3), only the
starting head differs.

| Starting head (accuracy before training) | Seed | Final holdout top-1 |
| --- | --- | --- |
| Linear probe (60.37%) | 0 | 71.37% |
| Linear probe (60.37%) | 1 | 71.03% |
| **Class prototypes (43.90%)** | 0 | **72.98%** |
| Class prototypes, head lr 1.2e-2 (30x) | 0 | 72.71% |

- **Starting from the prototypes is 1.6 to 2.0 points better**, although the
  linear-probe start is 16 points better before training. This is beyond the
  0.33-point seed noise.
- A likely explanation (not separately tested): the trained probe head is
  already confident (its temperature grew from 20 to about 65), so the loss and
  the gradients reaching the backbone are small early on, and the backbone
  adapts less. The prototype head starts at temperature 20 and pushes the
  backbone to adapt more, which is what in-distribution accuracy rewards.
  LP-FT is usually recommended for robustness to distribution shift, which this
  benchmark does not measure.
- From the prototypes, head lr 10x and 30x tie (72.98% vs 72.71%); 10x is kept.

### 4.6 Training length (prototype start, chosen recipe)

| Epochs (cosine fitted to each length) | Holdout top-1 per epoch | Final |
| --- | --- | --- |
| 1 (LP-FT start, for reference) | | 67.47% |
| 3 | 60.58, 70.17, 72.98 | **72.98%** |
| 6 | 58.11, 64.32, 68.27, 71.47, 73.66, 73.97 | **73.97%** |

Doubling from 3 to 6 epochs added about 1 point. In the 6-epoch run the holdout
loss rose slightly in the last epoch (1.285 to 1.302) while accuracy still rose,
so gains are slowing but have not reversed. A 12-epoch run is checking whether
the roughly 16 epochs of the Colab run start to overfit (section 9).

## 5. The final Colab run

- Config: `configs/full_ft_224px_proto-cosine_llrd0.75_top-lr4e-4_2.4h.yaml` (all settings below), launcher
  `notebooks/colab_van_train-one-config.ipynb` (branch `van-dev`), Colab A100 40 GB.
- Data: all 450,000 training images (no holdout).
- Training: 144 minutes, with the cosine schedule fitted to that limit.
  Expected about 16 epochs at about 830 img/s (throughput measured by
  Anastasia for this model on this GPU; the run's own epoch count is TBC).
- Prediction: `last.pt`. Plain center crop 0.9: **CodaBench 0.757** (ID
  963899). Same checkpoint with `--crop_pct 0.7 --tta_flip`: **CodaBench
  0.773** (ID 964799, 2026-10-06).
- Run folder, peak VRAM, GPU-hours and latency: to be added from the run folder
  on Drive.

The config file:

```yaml
inherits: shared_vit-b16_augreg2_defaults.yaml
method: full_ft
head_type: cosine
head_init: prototype
prototype_file: /content/drive/MyDrive/CS604_RUNS/prototypes.pt
lr: 4.0e-4            # top block; lower blocks get lr * layer_decay ** depth-from-top
layer_decay: 0.75
head_lr: 4.0e-3
weight_decay: 0.05
label_smoothing: 0.1
drop_path_rate: 0.1
clip_grad_norm: 1.0
batch_size: 256
grad_accum_steps: 1
num_workers: 8
epochs: 30            # only a ceiling; the time limit ends training
warmup: 1.5           # epochs = 5% of the 30-epoch ceiling = about 5% of the time budget
max_train_minutes: 144
schedule_fit_time: true
```

## 6. How to reuse this in your own runs

### 6.1 New trainer options (`src/train.py`, `src/model.py`)

All are off by default, so existing configs produce exactly the same run as
before. Add them to any config:

| Key | What it does | Value used here |
| --- | --- | --- |
| `layer_decay` | Layer-wise lr decay for ViTs: transformer block i (0 to 11) gets `lr * layer_decay ** (12 - i)`, the final norm keeps `lr`, the patch and position embeddings get `lr * layer_decay ** 13`; the classifier gets `head_lr` (or `lr` if unset). With 0.75 and lr 4e-4 the rates run from 4e-4 (top) to 9.5e-6 (embeddings). | 0.75 |
| `drop_path_rate` | Stochastic depth (timm's `drop_path_rate`, ramped linearly over the blocks). | 0.1 |
| `clip_grad_norm` | Clips the gradient norm before each optimizer step. | 1.0 |
| `schedule_fit_time` | Needs `max_train_minutes`. The warmup and cosine position follows the larger of the step count and the fraction of training time used, so the lr reaches zero exactly when the time limit stops training. Set `epochs` high (a ceiling) and `warmup` to about 5% of `epochs`. | true |

The trainer prints the rates it uses, for example
`[optim] layer_decay=0.75 top lr=0.0004 bottom lr=9.50e-06 head_lr=0.004`.

### 6.2 Prediction options (`src/predict.py`)

Both off by default (default output is unchanged):

- `--crop_pct 0.7`: evaluation center-crop fraction (the backbone's default is
  0.9).
- `--tta_flip`: averages the softmax of each image and its horizontal flip
  (doubles inference compute).

### 6.3 Running it on Colab

`notebooks/colab_van_train-one-config.ipynb` is a copy of `colab_train_one-config.ipynb` with
`BRANCH`, `METHOD` and `TEAM` set. Prediction finds the run folder training just
created, writes `val_team_08.txt/.zip` inside that folder (not the shared root),
also writes a crop 0.7 + flip submission to `tta_crop070_flip/`, then plots the
loss curves, measures latency, and (with `AUTO_DISCONNECT = True`) flushes Drive
and ends the runtime so it stops using compute units. Use Runtime > Run all on an
A100 40 GB. A 144-minute run writes about 0.4 GB to Drive.

### 6.4 Applying the findings to Task 2 (parameter-efficient methods)

Not tested yet, but the same pieces apply directly: the prototype-initialized
cosine head, a trained head with its own `head_lr`, warmup + cosine with
`schedule_fit_time`, label smoothing, and the prediction-time crop. Layer decay
matters less when only small adapters train.

## 7. Diagnosis that led to the head being trained

An earlier frozen-head experiment (LoRA rank 8 on attention qkv, cosine head
frozen at the prototypes) levelled off at 52.92% after 6.7 epochs. It
underfitted: only 61.6% on its own training images. The frozen head's
temperature (20) was too low (cross-entropy was lowest near 40), and features
stayed far from the frozen prototypes (mean cosine 0.58). Training the head
fixed this; with full fine-tuning the learned temperature rises on its own.

## 8. Prediction-time crop and flip (no retraining)

Re-scoring the laptop checkpoints on the internal holdout with different
evaluation crops, plus flip averaging:

| Eval center crop | 6 epochs | 6 epochs + flip | 3 epochs | 3 epochs + flip |
| --- | --- | --- | --- | --- |
| 1.0 | 72.77% | 73.13% | | |
| 0.95 | 73.55% | 73.87% | | |
| **0.9 (default)** | **73.97%** | 74.34% | **72.98%** | 73.36% |
| 0.875 | 74.42% | 74.68% | | |
| 0.85 | 74.60% | 75.00% | | |
| 0.8 | 75.04% | 75.44% | | |
| 0.75 | 75.34% | 75.54% | 74.31% | 74.55% |
| 0.7 | 75.38% | 75.50% | 74.57% | **74.85%** |
| 0.65 | 75.50% | **75.72%** | 74.44% | 74.81% |
| 0.6 | 75.14% | 75.39% | | |

- A tighter test crop is worth +1.4 to +1.6 points, flat between 0.65 and 0.75,
  at both lengths. The likely reason is the train/test object-size mismatch
  ("FixRes"): random-resized crops with scale 0.08 to 1 show objects larger in
  training than the 0.9 center crop does at test time.
- Flip averaging adds +0.1 to +0.4 on top, but doubles inference cost.
  Crop 0.7 alone gets most of the gain at no extra cost.
- Chosen: **crop 0.7 + flip** (+1.54 at 6 epochs, +1.87 at 3 epochs, +1.51 at
  12 epochs). The best crop stays at 0.65 to 0.7 from 3 to 12 epochs; the plain
  crop gain shrinks slowly with length (+1.58, +1.41, +1.23 points at 3, 6, 12
  epochs, 0.9 to 0.7).
- 12 epochs, plain / + flip: crop 0.9 74.47 / 74.74%; 0.8 75.44 / 75.62%;
  0.75 75.51 / 75.78%; 0.7 75.70 / **76.00%**; 0.65 75.84 / 76.01%; 0.6
  75.57 / 75.73%.
- **Confirmed on CodaBench:** re-predicting the final run's `last.pt` with
  `--crop_pct 0.7 --tta_flip` raised the score from 0.757 to **0.773** (+1.6,
  ID 964799), matching the holdout estimate. It took minutes, not a new
  training run.
- Averaging several crops (each with its flip) scores higher still on the
  holdout (six crops 0.6 to 0.9 at 12 epochs: 77.43%, vs 76.00% for crop 0.7 +
  flip), but costs about 6x the inference of a single crop; not adopted.

## 9. Open questions and next steps

- **12-epoch laptop run (done):** holdout 56.28, 62.28, 64.65, 66.44, 68.15,
  69.65, 70.85, 72.13, 73.25, 74.09, 74.40, **74.49%**: only +0.5 over 6
  epochs for twice the compute. Holdout loss was lowest at epoch 8 (1.377)
  and rose to 1.410 while accuracy crept up (more confident, not less
  accurate). Length is close to saturated for the ~16-epoch Colab run.
- **CodaBench with crop 0.7 + flip (done):** 0.773, up from 0.757.
- **More regularization in long runs** (drop_path 0.2, crop scale 0.35, weight
  averaging/EMA): the remaining lever on length; untested beyond 3 epochs.
- **Task 2** on the same recipe (section 6.4).

## 10. Glossary

- **Prototype head / class prototypes:** for each class, the average of the
  frozen backbone's normalized features over its training images; used as that
  class's classifier weight (with a cosine classifier). Gives 43.90% before any
  training.
- **Cosine classifier:** logits = temperature x cosine(feature, class weight).
  The temperature (starting at 20) is a trained parameter.
- **Linear probe:** training only the classifier head on frozen features.
- **LP-FT:** linear probe first, then fine-tune everything starting from that
  head. Here it was worse than starting from the prototypes.
- **Layer-wise lr decay:** lower transformer layers get smaller learning rates,
  so the early, general-purpose features change less.
- **Label smoothing 0.1:** the target puts 0.9 on the true class and spreads 0.1
  over all classes.
- **drop_path (stochastic depth):** randomly skips residual branches during
  training as regularization.
- **TTA (test-time augmentation):** predicting on modified copies of each test
  image (here the horizontal flip) and averaging.
- **Holdout top-1:** accuracy on Van's fixed 50,000-image internal holdout
  (section 3.2), not a CodaBench score.
