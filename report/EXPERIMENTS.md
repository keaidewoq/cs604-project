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

## Rules update 2026-10-03

The Prof confirmed that the classifier head must be trained and is not counted
in the 5% limit, and that any comparable-size ViT may be the backbone (see
`AGENTS.md`, "Course rules"). All entries below used a frozen or head-only
setup and form the frozen-head ablation. For new entries, give the backbone
name and pretraining data, and trainable parameters both without the head
(course number) and with it.

## Base-model screen: class-prototype accuracy (2026-10-03 to 2026-10-04)

**Purpose.** Choose the pretrained backbone before any adaptation. The Prof
asked us to start from the ImageNet-21k-only ViT-B/16 (`AGENTS.md`, rule 6) and
allows any vision transformer of comparable size (rule 3). We compared the two
ImageNet-21k ViT-B/16 releases, the IN21k→IN1k fine-tune used until now, and
DINOv2 ViT-B/14, with no training at all: each model's frozen features, a
cosine classifier built from class prototypes, and the same fixed holdout.

**Protocol (identical for every model).**

- Data: the course training set (450,000 labelled images, 45 per class,
  10,000 classes). The fixed internal holdout is 5 images per class (50,000
  images, `split_seed: 604`), the same file for every model (SHA-256
  `19a10c37a775559f4eee516215ad10c4b906ee50d66f7cbd95bc5ec88f2f2f69`).
- Prototypes: the remaining 40 images per class (400,000 images) go through the
  frozen backbone once with the model's own evaluation transform. Each class
  prototype is the mean of its 40 L2-normalized features (the pre-logits CLS
  token), normalized again. Built by `scripts/build_prototypes.py`.
- Scoring: each holdout image goes to the class with the highest cosine
  similarity to its feature; top-1 accuracy over 50,000 images.
- Confirmation: an evaluation-only run of the training code
  (`scripts/train.py`, `evaluate_only: true`) loads the prototypes into the
  cosine head and re-scores the holdout (bf16 backbone, float32 class scores).
  Its accuracy must match the prototype script's; it becomes the step-zero gate
  (`expected_initial_accuracy`) for training runs on that backbone.
- No optimizer updates. The prototypes use training labels, so this is
  supervised nearest-class-mean classification, not zero-shot.
- Hardware: RTX 5080 Laptop GPU, bf16 autocast for the backbone.

**Table 1. Candidate backbones.** Parameters exclude the classifier. GMACs are
multiply-accumulates per image at the evaluated input size, measured with
`torch.utils.flop_counter`.

| timm checkpoint | Pretraining data | Pretraining method | Params | Patch | Input | Patches | GMACs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `vit_base_patch16_224.orig_in21k` | ImageNet-21k (about 14M images, 21,843 classes) | Supervised, original ViT recipe (Dosovitskiy et al., 2021). The course PDF's `vit-base-patch16-224-in21k` | 85.80M | 16 | 224 | 196 | 16.85 |
| `vit_base_patch16_224.augreg_in21k` | ImageNet-21k | Supervised, AugReg recipe with stronger augmentation and regularization (Steiner et al., 2022) | 85.80M | 16 | 224 | 196 | 16.85 |
| `vit_base_patch16_224.augreg2_in21k_ft_in1k` | ImageNet-21k, then ImageNet-1k | `augreg_in21k` fine-tuned on ImageNet-1k (timm "augreg2") | 85.80M | 16 | 224 | 196 | 16.85 |
| `vit_base_patch14_dinov2.lvd142m` | LVD-142M (142M images retrieved from about 1.2B web images) | Self-supervised (Oquab et al., 2024); ViT-B distilled from DINOv2 ViT-g/14 (1.1B parameters) | 85.72M | 14 | 224 | 256 | 21.94 |

Evaluation transforms (each model's timm default): the three ViT-B/16 models
resize the short side to 248 and center-crop 224 (`crop_pct` 0.9), mean/std
0.5; DINOv2 resizes to 224 and center-crops 224 (`crop_pct` 1.0), ImageNet
mean/std. All use bicubic resizing. DINOv2's default input is 518 (1,369
patches, 116.98 GMACs, 6.9× the ViT-B/16 cost); it was run at 224 by resizing
its position embeddings (`image_size: 224`).

**Table 2. Results on the fixed 50,000-image holdout.** 95% intervals are
binomial (±1.96 standard errors). Differences between models are paired on the
same images, so the intervals overstate the uncertainty of a difference.

| Backbone | Prototype top-1 | Correct / 50,000 | 95% interval | vs IN1k fine-tune | Evaluation-only gate | Holdout loss at scale 20 |
| --- | --- | --- | --- | --- | --- | --- |
| `orig_in21k` | 41.816% | 20,908 | 41.38–42.25% | −2.084 | 41.816% | 3.045 |
| `augreg2_in21k_ft_in1k` (previous base) | 43.900% | 21,950 | 43.47–44.33% | — | 43.900% | 3.501 |
| `augreg_in21k` | 50.014% | 25,007 | 49.58–50.45% | +6.114 | 50.014% | 4.227 |
| **`dinov2` ViT-B/14 @ 224** | **53.758%** | **26,879** | 53.32–54.20% | **+9.858** | 53.758% | **2.602** |

- The training code's evaluation-only run reproduced every prototype score
  exactly (float32 class scores), so each backbone's step-zero gate equals its
  prototype accuracy.
- DINOv2 leads `augreg_in21k` by 3.744 points (1,872 images); `augreg_in21k`
  leads `orig_in21k` by 8.198 points. Both gaps are far larger than the
  intervals.
- The loss is cross-entropy with the cosine scale fixed at 20, reported only as
  a calibration hint: `augreg_in21k` has the highest loss despite its accuracy,
  so its features need a larger temperature. Training the head (and its
  temperature) addresses this.
- For reference, the previous best trained result, frozen-head rank-8 LoRA on
  the IN1k fine-tune after 6,250 updates (`prototype_lora_r8_fp32_screen`), was
  51.094%. DINOv2's prototypes alone exceed it.

**Table 3. Pretraining-data check (rule 4: no checkpoints trained on
iNaturalist 2021 or TreeOfLife-10M).**

| Backbone | iNaturalist or TreeOfLife in pretraining? | Source |
| --- | --- | --- |
| `orig_in21k`, `augreg_in21k` | No. ImageNet-21k only | timm model cards; ViT and AugReg papers |
| `augreg2_in21k_ft_in1k` | No. ImageNet-21k and ImageNet-1k | timm model card |
| `dinov2` | No iNaturalist dataset among LVD-142M's curation sources (ImageNet-22k, ImageNet-1k, Google Landmarks v2, fine-grained sets such as CUB-200, Flowers-102, Stanford Cars, and segmentation/depth/retrieval sets). iNaturalist 2018 and 2021 are marked as not used "as is" for pretraining, not used for retrieving pretraining data, and used for evaluation only. Because LVD-142M was retrieved from a pool of 1.2B web images, web copies of iNaturalist photos cannot be ruled out | DINOv2 paper, Tables 15 and 18 |

**Table 4. Provenance.** All prototype builds wrote exactly 40 images per class
and the holdout file above.

| Backbone | Prototype config | Prototype file (SHA-256, first 12) | Evaluation-only run | Commit |
| --- | --- | --- | --- | --- |
| `augreg2_in21k_ft_in1k` | `prototype_init.yaml` | `prototypes/class_prototypes.pt` (`c4c90ef9443c`) | `prototype-validate-fp32-head` | `010cf39` |
| `orig_in21k` | `prototype_init_orig_in21k.yaml` | `prototypes_orig_in21k/class_prototypes.pt` (`10238462a01e`) | `prototype_validate_orig_in21k` | `84b7902` |
| `augreg_in21k` | `prototype_init_augreg_in21k.yaml` | `prototypes_augreg_in21k/class_prototypes.pt` (`744ac61f816c`) | `prototype_validate_augreg_in21k` | `84b7902` |
| `dinov2` | `prototype_init_dinov2.yaml` | `prototypes_dinov2/class_prototypes.pt` (`60a22c99ef0f`) | `prototype_validate_dinov2` | `2f96526` |

Paths are under `cs604-outputs/`. Each run folder holds `baseline.json` (score,
loss, holdout and prototype hashes), `run.json` and `config.json`; each
prototype folder holds `class_prototypes.json` and its split files. The runs
record `git_dirty: true` only because `notebooks/colab_train.ipynb` was
uncommitted; the code was the stated commit. Build times (48–81 min) are not
comparable across models: until 2026-10-04 the laptop GPU was held at a 35 W
power limit by its charger, and the DINOv2 build switched to the full 175 W
partway through.

**Caveats.**

- Prototype accuracy measures frozen features with a nearest-class-mean head.
  It predicts, but does not guarantee, the ranking after LoRA and head training.
- Compute differs: at 224, DINOv2 costs 30% more per image than ViT-B/16, so
  under the fixed GPU-hour budget it sees fewer training images. A
  compute-matched setting (196 px, 196 patches) was not run.
- The evaluation crops differ (`crop_pct` 0.9 versus 1.0; each model's
  default). Their effect was not isolated.
- DINOv2's ViT-B has ViT-B size but was distilled from a 1.1B-parameter model.
- One measurement per model. The scores are deterministic up to GPU arithmetic
  (no training randomness); a paired significance test was not run because
  per-image predictions were not saved.
- DINOv2 with registers (`vit_base_patch14_reg4_dinov2.lvd142m`) and other
  ViT-B families (CLIP/SigLIP, EVA-02) were not screened.

**Decision.** Pending. DINOv2 ViT-B/14 at 224 has the best starting point
(53.758%). Rule 6 asks to start from the ImageNet-21k ViT-B/16; if that holds,
`augreg_in21k` (50.014%) is the base and DINOv2 is reported as a comparison.

## Completed

- Prototype LoRA rank-8, full time budget, frozen head, finished 2026-10-03 00:21 SGT: internal top-1 **52.924%** (26,462 / 50,000), from a remeasured start of **43.934%**. Per-epoch holdout: 49.74, 50.92, 51.77, 52.72, 52.70, 52.92, 52.91% (the last is a partial epoch at the budget stop). Gain over the 1-epoch run (51.826%): +1.10 points for 10× the budget. No public score.
  Config: `configs/experiments/prototype_lora_r8_fp32_budget_40.yaml`. Commit: `8cb47324cb8cc4bfb651356809c923b2c79ad471`, clean.
  Method: as `prototype_lora_r8_fp32_40` (frozen 40-image cosine prototypes, fp32 class scores, rank-8 `qkv` LoRA, lr 0.0005, batch 32), with `budget_48gb_hours: 1.9` and `epochs: 30` as an upper bound, so the cosine schedule followed the time budget. `num_workers` 8 instead of 4.
  Changed variable: training length (83,308 updates, 6.7 epochs, versus 12,500 updates, 1 epoch).
  Total parameters: 93,773,569; trainable LoRA: 294,912 (0.3145%); trainable classifier: 0.
  NVIDIA A100-SXM4-80GB (79.25 GiB); peak allocated 2.0887 GiB; elapsed 4144.71 s, GPU-hours 1.1513, normalized 48-GiB hours 1.9009; `stopped_by_budget: true`. Training about 740 images/s, holdout evaluation about 910 images/s; the 8 holdout evaluations used about 11% of the budget. Source: that run's `run.json` and `metrics.csv`.
  Diagnosis (evaluation-only, `analysis/diagnose_best.json` in the run folder): 61.55% top-1 on a 2-per-class sample of its own training images (underfitting); cross-entropy lowest near cosine scale 40 versus the frozen 20; mean cosine to the true prototype 0.58.
  Conclusion: training length is not the bottleneck for the frozen-head setup. Next: train the head (now required), then screen backbones. See `report/HANDOFF.md`.

- Prototype LoRA rank-8 screen, 2026-09-30: internal top-1 **51.094%** (25,547 / 50,000) after two epochs, up from a remeasured start of **43.900%** (21,950 / 50,000). Gain versus the 43.90% prototype baseline: **+7.194 percentage points** (3,597 images). Epoch 1 was 50.508% (25,254 / 50,000); epoch 2 added 0.586 points. No public score.
  Config: `configs/experiments/prototype_lora_r8_fp32_screen.yaml`. Recorded commit: `910b2ff4d3cfcf89485243550fd429277d50aa95` on `prototype-baseline` (`git_dirty: true` because this config was uncommitted during the run).
  Method: frozen 40-image cosine prototypes, float32 class scores, LoRA rank 8 on attention `qkv` only (`train_head: false`). Ten images per class for the optimizer, so this is not a 10-shot study.
  Changed variable: 6,250 AdamW updates of the LoRA adapters (lr 0.0005, weight decay 0.01, label smoothing 0, batch 32, 2 epochs).
  Checkpoints: `cs604-outputs\prototype-lora-r8-fp32-screen\baseline.pt` (step 0, accuracy 0.439) and `best.pt` / `last.pt` in the same directory (step 6250, accuracy 0.51094). No prediction file.
  Total parameters: 93,773,569; trainable LoRA: 294,912 (0.3145%); trainable classifier: 0.
  RTX 5080 Laptop GPU, 15.92 GiB device capacity; peak allocated 2.1347 GiB; logged elapsed 7351.8314 s (2h 2m 32s), GPU-hours 2.0421754, normalized 48-GiB hours 0.6773215.
  Initial holdout throughput 69.4753 images/s; training about 42 images/s in the second epoch; final holdout throughput 91.6021 images/s at evaluation batch size 64. Source: that run's `run.json`, `baseline.json`, and `metrics.csv`.
  The step-zero gate passed (`accuracy_delta` 0.0). Both epochs beat the start, and epoch 2 beat epoch 1, so `best.pt` is the epoch-2 model. This is above the head-only screen (45.858%) and under the 5% parameter budget. The original prototypes and earlier checkpoints were left in place.

- Prototype head-only screen, 2026-09-30: internal top-1 **45.858%** (22,929 / 50,000) after one epoch, up from a remeasured start of **43.900%** (21,950 / 50,000). Gain versus the 43.90% prototype baseline: **+1.958 percentage points** (979 images). No public score.
  Config: `configs/experiments/prototype_head_screen.yaml`. Commit: `929f8f361007a3ba90e08a21d6cd2b7592f12c08` on `prototype-baseline` (`git_dirty: false`).
  Method: frozen ViT-B/16 backbone, cosine head trained from the validated prototype checkpoint, with `cosine_head_fp32: true`. Five images per class for the optimizer; the initializer itself used 40 images per class, so this is not a five-shot result. The head is a diagnostic baseline, not the under-5% adaptation method.
  Changed variable: 782 AdamW updates of the classifier only (lr 0.001, weight decay 0, label smoothing 0, batch 64).
  Checkpoints: `cs604-outputs\prototype-head-screen\baseline.pt` (step 0, accuracy 0.439) and `best.pt` / `last.pt` in the same directory (step 782, accuracy 0.45858). No prediction file.
  Total parameters: 93,478,657; trainable classifier: 7,680,001 (8.2158%); trainable backbone: 0.
  RTX 5080 Laptop GPU, 15.92 GiB device capacity; peak allocated 1.1328 GiB; logged elapsed 1606.8395 s (26m 47s), GPU-hours 0.4463443, normalized 48-GiB hours 0.1480375.
  Initial holdout throughput 157.8886 images/s; training about 80 images/s after warmup; final holdout throughput 79.1746 images/s at evaluation batch size 128. Source: that run's `run.json`, `baseline.json`, and `metrics.csv`.
  The step-zero gate passed (`accuracy_delta` 0.0). Training improved the fixed 50,000-image holdout, so `best.pt` is the trained head. `baseline.pt` and the earlier `prototype-validate-fp32-head` checkpoints were left in place. Next training run needs a new experiment YAML and output directory.

- Prototype starting-point validation, 2026-09-30: **43.900%** internal top-1
  (21,950 / 50,000) using the original holdout; **zero optimizer updates**.
  Config: `configs/experiments/prototype_validate_fp32_head.yaml`.
  Checkpoints: `cs604-outputs\prototype-validate-fp32-head\baseline.pt`
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

## Experiment direction updated 2026-10-02

See [the iNaturalist 2021 strategy](INAT2021_STRATEGY.md) for dataset research,
verified newer run evidence, and the ordered experiment plan. The current saved
reference is **51.826% internal top-1** from
`cs604-outputs/prototype_lora_r8_fp32_40/run.json`; its prediction audit identifies
the same selected checkpoint. This is not a recorded public leaderboard score.

Prioritize a prototype-initialized full fine-tune, a controlled crop-policy
ablation, and then image-detail/inference tests. Reconcile cumulative compute
before scheduling runs. Broad LoRA rank and last-block sweeps are deferred.

## Earlier comparison backlog (superseded in priority)

- Frozen backbone with trained linear head
- Full fine-tuning
- Selected final transformer blocks
- LoRA ranks 4, 8, and 16
- Frozen prototype head with LoRA
- Winning method with and without each final loss/augmentation change

Do not reconstruct these values at report time. `run.json`, `metrics.csv`, and
the `.inference.json` beside each prediction are the evidence for the report.
