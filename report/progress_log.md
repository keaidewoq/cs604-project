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

### 2026-10-06 — Van — full FT recipe tuned on a laptop (`configs/van_full_ft_proto_llrd.yaml`)

- run_dir: laptop screens in Van's own pipeline (RTX 5080 laptop); the Colab run of this config (A100 40 GB, `notebooks/van_colab_launcher.ipynb`, branch `van-dev` at f5d654a) is under `CS604_RUNS`, folder name TBC
- val acc (ours / CodaBench): internal holdout (5 training images/class held out, trained on the other 40) **72.98%** after 3 epochs, **73.97%** after 6 / **CodaBench 0.757** (ID 963899, 2026-10-06; all 45 images/class, 144 training minutes, plain center crop 0.9, no test-time augmentation)
- trainable % / peak VRAM / GPU-hours: 100% / 5.8 GB at batch 64 / 6 epochs = 2.1 laptop hours (about 0.7 of the 48 GB-hour budget)
- what changed vs `full_ft_proto.yaml`: prototype-initialized **cosine** head; backbone lr 4e-4 with **layer-wise lr decay 0.75** (block i gets lr x 0.75^(12 - i)); head lr 4e-3; 5% warmup then cosine to zero **fitted to the time limit**; label smoothing 0.1; drop_path 0.1; gradient clipping 1.0; batch 256.
- findings (one change at a time, 3 epochs): without layer decay 63.1% (vs 69.8%); head lr 10x the backbone +1.5 points; prototype start beats first training a linear probe (LP-FT) by 1.6–2.0 points; crop scale, colour jitter and head lr 30x no difference; seed-to-seed spread 0.33 points.
- trainer additions (all off by default, so existing configs run exactly as before): `layer_decay`, `drop_path_rate`, `clip_grad_norm`, `schedule_fit_time` (with `max_train_minutes`, the cosine reaches zero at the time limit instead of being cut off mid-decay).
- next: fill in epochs reached, peak VRAM, GPU-hours and latency from the run folder. Re-predict the same `last.pt` with a center crop of 0.7 plus flip averaging (`--crop_pct 0.7 --tta_flip`), which added +1.5 to +1.9 points on the internal holdout; no retraining needed.

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
- val acc (ours / CodaBench): **0.719** holdout top-1 (best epoch 6: 0.7194; final 0.7192), top-5 0.885 / CodaBench **0.717**
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
### 2026-10-07 — Anastasia — LoRA r24 + extras @ 336 with Van'srecipe (`configs/lora_r24_336_recipe.yaml`)

- run_dir: `CS604_RUNS/lora_20261007_051926` (launcher `notebooks/colab_launcher_ana_lora336.ipynb`, `ana-branch`)
- val acc (ours / CodaBench): no internal holdout (trained on all 45 images/class) / **0.785** plain center crop 0.9; **0.793**with crop 0.7 + flip TTA
- trainable % / peak VRAM / GPU-hours: **4.65% of the backbone** (4.00M / 85.99M, head excluded) / 19.40 GB / 2.40 (144.0 training min, time limit reached; A100 40 GB)
- what changed vs the 0.717 LoRA r16 run: rank 16 → 24; also trained LayerNorms, all biases, pos_embed and cls_token (0.46M, noweight decay); prototype cosine head (`prototypes_336.pt`) instead of a linear-probe head; head lr 1e-4 → 3e-3 (10× the LoRA lr); effective batch 128 → 256; weight decay 1e-4 → 0.05. Unchanged: LoRA on qkv/proj/fc1/fc2, LoRA lr 3e-4, alpha = rank, label smoothing 0.1, 5% warmup + cosine fitted to 144 min.
- results: **+6.8 points over the 0.717 LoRA**; within 2.2 points of full FT @336 (0.807) with 4.65% of the backbone trained; above Van's full FT @224 (0.757) and Qian's last-block attention (0.668, 2.75%). Several changes at once, so the gain is not yet attributed to any one of them. **~6.8 epochs** (~1,270 s/epoch, ~355 img/s) vs ~8.5 for full FT @336: LoRA was ~25% slower per epoch than full FT (many small unmerged matmuls) and used nearly the same peak memory (19.40 vs 20.05 GB), so here it saved parameters, not training time or VRAM.
- cost: latency measured **unmerged 14.44–14.98 ms/image at batch 1** (1.07 ms/image at batch 64): +89% vs full FT, because the 48 LoRA layers run as ~96 extra small matmuls. **Merged** into the base weights (`scripts/latency_merged.py`, `inference_latency_merged.json`, same A100 session): **7.68 ms at batch 1, 0.650 ms at batch 64**, the same as a plain ViT-B with the same head (7.58 / 0.651 ms), i.e. no inference cost. Merge check: 0 LoRA layers left, fp32 max logit difference 1.5e-5, 100% fp32 top-1 agreement (bf16 96.5% on synthetic noise inputs, from rounding at near-ties). Submitted predictions used the unmerged model. Crop 0.7 + flip needs two forward passes.
- trainer additions (all off by default): `lora_targets` (qkv, proj, fc1, fc2), `unfreeze_extra` (norm, bias, pos_embed, cls_token; no weight decay on these), and a per-run `[params]` line giving trainable % of the original backbone with the head excluded(repo default LoRA r8 on qkv prints 0.34%, matching the brief).
- next: ablate the changes (extras, rank, head init, head lr) at pilot scale; Adapter (width 96) and AdaptFormer (width 192) atthe same ~4.7% budget and setup.

### 2026-10-09 — Qian — prefix tuning ablation test

- run_dir: `CS604_RUNS/qq_prefix_ablation_336_seed42_v1/runs/screen/` (`qq_prefix_tuning_ablation`; `colab_launcher_qq.ipynb`).
- val acc (internal): best **0.71688** (`all_p50`). All arms' best accuracy occurred at epoch 5; one seed, convergence unconfirmed.
- trainable % / peak VRAM / GPU-hours: selected arm **1.071792%** backbone (classifier excluded) / **7.97 GiB** / **1.188 wall GPU-hours**, including **64.80 training minutes** on A100 40GB. All six runs below.
- what changed: frozen augreg2 at 336 with fit-only prototype linear head; 40 fit + 5 holdout images/class, seed 42. Five epochs/arm; AdamW prefix/head LR 1e-3, weight decay 0, batch 64, BF16, 5% warmup + cosine. Compared placements at length 5, then lengths 20/50 at all blocks. `all_p50` adds **8.36 points** over baseline; its **0.04-point** lead over `all_p5` is inconclusive with one seed.
- next: Qian selected **all blocks, length 50**. Rebuild all-data prototypes and train from pretrained weights on all 450,000 images for **144 training minutes**, then evaluate CodaBench (planned 2026-10-10).

| Arm | Holdout top-1 | Backbone trainable % | Peak VRAM (GiB) | Training min | Wall GPU-hours |
|---|---:|---:|---:|---:|---:|
| Classifier only | 0.63328 | 0 | 1.24 | 61.86 | 1.141 |
| Last block, length 5 | 0.65136 | 0.008932 | 1.45 | 61.40 | 1.130 |
| Last 4 blocks, length 5 | 0.68996 | 0.035726 | 3.68 | 61.89 | 1.139 |
| All blocks, length 5 | 0.71648 | 0.107179 | 8.35 | 63.79 | 1.170 |
| All blocks, length 20 | 0.71634 | 0.428717 | 7.90 | 65.40 | 1.202 |
| All blocks, length 50 | **0.71688** | 1.071792 | 7.97 | 64.80 | 1.188 |

### 2026-10-10 — Qian — configs/prefix_tuning_336.yaml

- run_dir: `CS604_RUNS/prefix_20261010_000404_id974631` (`last.pt`; `qq_branch_v2`).
- val acc (ours / CodaBench): **0.71688** (selection ablation, `all_p50`) / **0.721** (final run), submission **974631**.
- trainable % / peak VRAM / GPU-hours: **1.071792%** backbone (classifier excluded) / **7.97 GiB** / **2.402807 wall GPU-hours**; 144 training minutes on A100 40GB. Latency: **9.644 ms/image**, batch 1, BF16 synthetic forward-only.
- what changed: final prefix-tuning run at 336, all blocks with length 50; prototype-initialized linear head using `prototypes_336.pt`; batch 64, prefix/head LR 1e-3, weight decay 0, seed 42; 5% time warmup + cosine, no epoch ceiling.

### 2026-10-09 — Anastasia — Adapter w96 + extras @ 336 with Van's recipe (`configs/adapter_w96_336_recipe.yaml`)

- run_dir: `CS604_RUNS/adapter_20261009_131150`; lr pilot `CS604_RUNS/pilot_adapter_336_20261009_125653` (launcher `notebooks/colab_launcher_ana_peft336.ipynb`, `ana-branch`)
- val acc (ours / CodaBench): no internal holdout (trained on all 45 images/class) / **0.776** plain center crop 0.9; **0.785** with crop 0.7 + flip TTA
- trainable % / peak VRAM / GPU-hours: **4.68% of the backbone** (4.02M / 85.99M, head excluded) / 15.54 GB / 2.40 (144.0 training min, time limit reached; A100 40 GB)
- what it is: 24 serial bottleneck adapters (after attention and after the MLP in all 12 blocks, 768 → 96 → 768, ReLU, up-projection zero-initialised) plus LayerNorms, all biases, pos_embed and cls_token (no weight decay), sized to the LoRA r24 budget. Same setup as the LoRA r24 run: prototype cosine head, head lr 3e-3, weight decay 0.05, label smoothing 0.1, clip 1.0, effective batch 256, 5% warmup + cosine fitted to 144 min.
- lr pilot (1,000 classes, 3 epochs, holdout 5/class): 3e-4 / 1e-3 / 3e-3 = 0.8938 / **0.8958** / 0.8840, no early dips. Best lr is inside the sweep; the tie rule (within 0.3 points → lower lr) chose **3e-4**. Consistent with Van's AdaptFormer sweep (1e-3 ≈ 3e-3 best).
- results: **0.9 points below LoRA r24** at the same budget, both plain (0.776 vs 0.785) and with crop + flip (0.785 vs 0.793), despite more epochs (**~9.1** vs ~6.8). Crop + flip added +0.9, the largest TTA gain so far (full FT +0.6, LoRA +0.8).
- cost: fastest training of all 336 runs: **~951 s/epoch (~473 img/s)** vs 1,017 s full FT and 1,272 s LoRA; lowest peak VRAM (15.54 vs 20.05 GB full FT). Adapters cannot be merged into the weights: latency **11.60 ms/image at batch 1 (+46% vs full FT's 7.94)**, 0.73 ms/image at batch 64 (+15%). LoRA, once merged, has no extra latency (7.68 ms), so Adapter is cheaper to train but slower to run.
- next: AdaptFormer covered by Van's run (0.779 / 0.790, 4.13%). Try Van's CosFace margin (0.1) on LoRA r24; measure merged-LoRA latency.

### 2026-10-12 — Anastasia — LoRA r24 recipe ablation + DoRA, pilot scale (`configs/ablation_lora_r24.yaml`)

- run_dir: `CS604_RUNS/ablation_ablation_lora_r24/` (`scripts/ablation.py`, launcher `notebooks/colab_launcher_ana_ablation.ipynb`, `ana-branch`); arms A–F on 2026-10-11, G–I on 2026-10-12
- setup: one change at a time from `lora_r24_336_recipe.yaml`; 1,000-class pilot subset (seed 0), 40 fit images/class, 3 epochs (5% warmup + cosine), scored on a labelled 5/class holdout; one seed per arm. Both starting heads rebuilt from the fit images only (pilot-class prototypes recomputed; linear probe trained on the same images), so no head has seen the holdout. Not ablated: batch 256 and weight decay 0.05 (kept in all arms).

| Arm | Change | Holdout top-1 | vs A (points) | Backbone trainable | Pilot min |
|---|---|---:|---:|---:|---:|
| A | full recipe | **0.8984** | — | 4.65% | 6.3 |
| B | no norms / biases / pos_embed / cls_token | 0.8970 | −0.14 | 4.12% | 5.9 |
| C | rank 16 instead of 24 | 0.8918 | −0.66 | 3.28% | 6.3 |
| D | linear head from a linear probe (LP-FT) | 0.8692 | −2.92 | 4.65% | 6.3 |
| E | head lr 1e-4 instead of 3e-3 | 0.8370 | −6.14 | 4.65% | 6.3 |
| F | B + C + D + E (≈ the 0.717 run's setup) | 0.8538 | −4.46 | 2.74% | 6.0 |
| G | D with head lr 1e-4 (probe head, slow lr) | 0.8572 | −4.12 | 4.65% | 6.5 |
| H | linear head started from prototypes | 0.8892 | −0.92 | 4.65% | 6.3 |
| I | DoRA instead of LoRA (extension) | 0.8996 | +0.12 | 4.75% | 7.7 |

- results: **the head setup drives most of the gain.** Head start × head lr: prototype cosine head 0.8984 (fast) / 0.8370 (slow); linear-probe head 0.8692 (fast) / 0.8572 (slow). A fast head lr helps both starts, but the prototype start depends on it 5× more (−6.1 vs −1.2 points): it starts as class means and must be trained into a classifier, while a probe head is already one. **The prototype start wins only with a fast head lr** (+2.9 fast, −2.0 slow), so the two must be chosen together; the 0.717 run's combination (probe + slow) was not even the best for a probe head (+1.2 with a fast lr). Of the 2.9-point gap between the prototype cosine head (A) and LP-FT (D), ~2.0 comes from the prototype start (H − D) and ~0.9 from the cosine head (A − H). Rank 24 over 16: +0.66. Extras (norms, biases, embeddings): +0.14, within seed noise (~0.3), so dropping them saves 0.53% of the backbone at no measurable cost. F's −4.5 covers most of the full-scale gap (0.717 → 0.785); the rest is plausibly scale plus batch size and weight decay.
- DoRA (I): +0.12 points (a tie) at +0.10% parameters, but **22% slower** to train (7.7 vs 6.3 min). Plausibly rank 24 already gives LoRA enough capacity (DoRA's reported gains are largest at low rank), and the head, not the adapter, is the bottleneck here. DoRA merges like LoRA (`lora_dora`, tested: merged = unmerged). Confirmed at full scale in the next entry.
- limitations: pilot scale (absolute accuracies inflated by the easier 1,000-class task; use for direction and ranking); one seed per arm; G's best epoch (0.8588) was slightly above its final (0.8572).
- next: report the head start × head lr table and the A/H/D split; final PEFT model stays LoRA r24 (optionally without extras, 4.12%).

### 2026-10-13 — Anastasia — DoRA r24 + extras @ 336 with Van's recipe (`configs/lora_r24_336_dora.yaml`)

- run_dir: `CS604_RUNS/lora_20261012_130611` (launcher `notebooks/colab_launcher_ana_dora336.ipynb`, `ana-branch`)
- val acc (ours / CodaBench): no internal holdout / **0.783** plain (submission 983071); **0.793** with crop 0.7 + flip (submission 983064)
- trainable % / peak VRAM / GPU-hours: **4.75% of the backbone** (4.08M / 85.99M, head excluded; +82,944 magnitude params vs LoRA) / **28.06 GB** / 2.40 (144.0 training min, time limit reached; A100 40 GB)
- what changed vs `lora_r24_336_recipe.yaml` (0.785): only `lora_dora: true`. DoRA (Liu et al., ICML 2024) splits each adapted weight into a direction (updated by LoRA) and a learned per-output magnitude, closer to how full FT updates weights.
- results: **tie with LoRA** (−0.2 plain, ±0 with crop + flip), matching the pilot (arm I: +0.12). DoRA trained **25% slower** (~1,585 vs ~1,272 s/epoch) and reached only **~5.45 epochs** vs ~6.8, so it matched LoRA with ~20% fewer epochs, but at a fixed 144-min budget the slowdown cancels the per-epoch gain. Peak memory +45% (28.06 vs 19.40 GB): each step builds the full adapted weight of all 48 layers to compute its norms. Crop + flip added +1.0, the largest TTA gain so far. Consistent with the ablation: rank 24 already gives LoRA enough capacity, and the head setup, not the adapter, limits accuracy here.
- cost: latency measured **unmerged 23.14–24.36 ms/image at batch 1** (1.41 ms/image at batch 64), the slowest of all runs, because every forward pass recomputes the weight norms of all 48 layers. **Merged** (`scripts/latency_merged.py`, same A100 session): **7.71 ms at batch 1, 0.649 ms at batch 64**, identical to a plain ViT-B with the same head (7.71 / 0.651 ms), so DoRA, like LoRA, has no inference cost once merged. (A second merged-latency session on the LoRA r24 checkpoint gave 8.35 ms vs 8.20 ms plain; the first gave 7.68 vs 7.58. Merged stays within ~0.15 ms of plain both times.)
- next: final PEFT model stays LoRA r24 (same accuracy, faster and lighter to train).

### 2026-10-16 — Anastasia — LoRA from block k up + progressive resizing, pilot scale (`configs/ablation_lora_depth_res.yaml`)

- run_dir: `CS604_RUNS/ablation_ablation_lora_depth_res/` (`scripts/ablation.py`, launcher `notebooks/colab_launcher_ana_depth_res.ipynb`, `ana-branch`)
- setup: same pilot as the r24 ablation (1,000 classes, 40 fit / 5 holdout per class, prototype heads rebuilt from fit images), A = `lora_r24_336_recipe.yaml`. Two new opt-in options: `first_trainable_block: k` (LoRA only in blocks k..11; norms, biases, `pos_embed` and `cls_token` below k frozen, so backward stops at block k) and `low_res_size: 224` / `low_res_fraction: 0.67` (train at 224 for the first 67% of progress, then 336; eval and inference at 336). J–M: same 3 epochs as A (accuracy cost). `*_t` arms: same training minutes as A (5.75 min, via `max_train_minutes: {match: A_full_recipe}`), i.e. what the speed-up buys under a time budget.

| Arm | Change | Holdout top-1 | vs A | Trainable | Epochs | Speed-up | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| A | full recipe | **0.8986** | — | 4.65% | 3 | 1.00× | 19.40 GB |
| J | blocks 4–11, r40 | 0.8894 | −0.92 | 4.67% | 3 | 1.29× | 13.59 GB |
| K | blocks 6–11, r54 | 0.8830 | −1.56 | 4.70% | 3 | 1.43× | 10.64 GB |
| L | 224 → 336 | 0.8866 | −1.20 | 4.65% | 3 | 1.55× | 19.39 GB |
| M | J + L | 0.8878 | −1.08 | 4.67% | 3 | 1.87× | 13.59 GB |
| J_t | J, A's minutes | 0.8890 | −0.96 | 4.67% | 4 | 1.33× | 13.59 GB |
| L_t | L, A's minutes | 0.8898 | −0.88 | 4.65% | 6 | 2.00× | 19.39 GB |
| M_t | M, A's minutes | 0.8872 | −1.14 | 4.67% | 7 | 2.33× | 13.59 GB |

- results: **neither speed-up helps, even at equal compute.** All seven arms lost 0.9–1.6 points (beyond ~0.3 seed noise, all in the same direction). Extra epochs bought almost nothing (L 3 epochs 0.8866 vs L_t 6 epochs 0.8898; M_t with 7 epochs still −1.1), so at pilot scale LoRA plateaus within ~3 epochs and the time budget is not the bottleneck. Training at 224 costs ~1.2 points and a final phase at 336 doesn't recover it. The speed-ups and VRAM savings are real on the A100 (J 13.6 GB, K 10.6 GB vs 19.4 GB), useful only on a memory-limited GPU.
- caveat on interpretation: J and K also froze `pos_embed` + `cls_token` (~0.44M, most of A's extras) and concentrated the same budget in fewer blocks, so their loss does **not** show that the early blocks specifically matter. The param-selection pilot (next entry) found no depth effect, which points to concentration rather than depth. A control run (`configs/ablation_lora_depth_control.yaml`: fixed r24 on fewer blocks, and J with `pos_embed` trainable) is set up to separate the two; results TBC.
- limitations: pilot scale (1,000 classes may plateau sooner than 10,000); one seed per arm.
- next: keep A as the LoRA submission; no lr / `low_res_fraction` sweep or full run.

### 2026-10-17 — Anastasia — parameter-selection pilot: depth, fine-tuned angle, gradient at equal budget (`configs/ablation_param_selection.yaml`)

- run_dir: `CS604_RUNS/ablation_ablation_param_selection/` (`scripts/ablation.py` + `scripts/select_params.py`, launcher `notebooks/colab_launcher_ana_param_selection.ipynb`, `ana-branch`)
- question: at a fixed budget of ~4.6% of the backbone, which parameters should be trained? Same pilot as above (1,000 classes, 40 fit / 5 holdout per class, 3 epochs, one seed per arm). Directly trained pretrained weights use their own `unfreeze_lr` (3e-5 for whole matrices, 1e-4 for GPS entries; the latter a guess, not swept).
- arms: **A** LoRA r24 everywhere (current recipe). **F** full FT at pilot scale, Van's LLRD recipe (`F_llrd`) and one flat lr 3e-5 (`F_flat`). **S0 / S5 / S11** (surgical, Lee et al. ICLR 2023): block *i*'s full attention (qkv + proj, 2.36M) + LoRA r8 elsewhere. **Y** (partial FT, Ye et al. 2023): rank matrices by angle between pretrained and `F_flat` weights; `Y_layer` = top whole attention layer + LoRA r8, `Y_matrix` = top single matrices filled to budget, no LoRA. **G** (GPS, Zhang et al. CVPR 2024): top-K weights per output neuron by accumulated gradient on fit images (cross-entropy, not GPS's contrastive loss, since the head starts from prototypes), K = 42 and K = 10. Y and G selections use only fit images (no holdout leakage).

| Arm | Holdout top-1 | vs A | Backbone trainable | Peak VRAM | Training min |
|---|---:|---:|---:|---:|---:|
| A: LoRA r24 everywhere | **0.8996** | — | 4.65% | 19.40 GB | 5.86 |
| F_llrd: full FT, LLRD | 0.8960 | −0.36 | 100% | 20.05 GB | 4.75 |
| F_flat: full FT, flat lr | 0.8956 | −0.40 | 100% | 20.05 GB | 4.63 |
| G_k42: GPS | 0.8944 | −0.52 | 4.63% | 20.37 GB | 4.64 |
| Y_layer: block 11 attention + LoRA r8 | 0.8908 | −0.88 | 4.61% | 19.30 GB | 5.78 |
| S5: block 5 attention + LoRA r8 | 0.8900 | −0.96 | 4.61% | 19.30 GB | 5.80 |
| S11: block 11 attention + LoRA r8 | 0.8894 | −1.02 | 4.61% | 19.30 GB | 5.77 |
| G_k10: GPS | 0.8892 | −1.04 | 1.51% | 20.37 GB | 4.64 |
| S0: block 0 attention + LoRA r8 | 0.8888 | −1.08 | 4.61% | 19.30 GB | 5.78 |
| Y_matrix: 4 single matrices, no LoRA | 0.8738 | −2.58 | 4.65% | 13.51 GB | 4.06 |

- selections: `Y_layer` picked `blocks.11.attn.qkv` + `blocks.11.attn.proj`, the same as S11, so the two are identical runs; their 0.14-point difference (0.8908 vs 0.8894) is a measured run-to-run noise estimate (GPU nondeterminism), consistent with the ~0.3 tie threshold. `Y_matrix` picked `blocks.11.attn.qkv` and `attn.proj` of blocks 0, 9, 10: 4 of 49 matrices, no MLP, because the larger `fc1` matrices next in the ranking didn't fit the remaining budget.
- angles (`F_flat`, degrees between pretrained and fine-tuned weights, `angles.csv`): all 49 matrices between 0.79° (`patch_embed`, the lowest) and 1.22° (`blocks.11.attn.qkv`). `qkv` (0.92 → 1.22) and `fc1` (0.90 → 1.16) rise steadily with depth; `proj` and `fc2` are mostly flat (block-0 `proj` outlier at 1.16°). Full FT changes every layer a little and the high-level layers most.
- results: **spreading the budget beats concentrating it.** The two best ways to spend ~4.6% (LoRA everywhere, GPS) touch every layer; one full-rank attention block costs ~1 point at any depth, and the most concentrated arm (`Y_matrix`) loses 2.6. **Depth makes no difference** (S0 ≈ S5 ≈ S11, within noise), so the early-layer hypothesis from Lee et al. is not supported here; with the angles (patch embedding moved least), iNat vs ImageNet is not a low-level appearance shift. How much a layer moves under full FT doesn't predict how useful it is to tune at a fixed budget. G_k10 is the best accuracy per parameter (−1.0 at 1.51%), but GPS saves parameters, not compute: full gradients are still computed and masked (20.37 GB, no speed-up).
- caveat: **full FT does not beat LoRA at pilot scale** (−0.4), unlike full scale (0.807 vs 0.785). The pilot (40k images, 1,000 classes, 3 epochs) can rank PEFT variants but can't show the capacity limit that makes full FT win at 450k images; all "vs F" conclusions are pilot-only. Consistent with a data-size crossover, but the settings also differ in class count and epochs. Angle comparisons across matrix types are weaker than within a type (angle is relative to ‖W‖). Pilot angles come from 3 epochs at lr 3e-5, so all changes are small.
- next: depth-control run (`configs/ablation_lora_depth_control.yaml`) to test "spread vs concentrated" directly (C16_all vs J24_embed at ~3.2%) and the `pos_embed` confound (J40 vs J40_embed); results TBC. Angle-vs-depth figure for the report.

### 2026-10-11 — Qian — BitFit ablation test

- run_dir: `CS604_RUNS/qq_bitfit_ablation_336_subset_seed42_v2/` (`qq_bitfit_ablation`; `colab_launcher_qq.ipynb`). Selected arm reused from `runs/pilot/all_biases_lr0.001/seed42/`; other arms under `runs/screen/`.
- val acc (internal subset / CodaBench): **0.8810** (`all_biases`) / not submitted. All arms' best accuracy occurred at epoch 3; one seed, convergence unconfirmed.
- trainable % / peak VRAM / GPU-hours: selected arm **0.119683%** backbone (classifier excluded) / **7.01 GiB** / **0.074 wall GPU-hours**, including **4.02 training minutes** on A100-SXM4-40GB.
- what changed: augreg2 at 336, fit-only prototype linear head; same 1,000-class subset, 40 fit + 5 holdout images/class, seed 42. Three epochs/arm; AdamW bias/head LR 1e-3, weight decay 0, batch 64, BF16, 5% warmup + cosine. Compared bias scopes after an all-bias LR pilot; matching pilot run reused. All biases adds **6.18 points** over head-only and **0.64 points** over attention biases. Subset scores are not directly comparable to full-scale results.
- next: selected **all biases**, bias/head LR **1e-3**. Returned to `qq_branch_v2`; rebuild all-data prototypes at 336 and train from pretrained weights on all 450,000 images for **144 training minutes**, then evaluate CodaBench.

| Arm | Holdout top-1 | Backbone trainable % | Peak VRAM (GiB) | Training min | Wall GPU-hours |
|---|---:|---:|---:|---:|---:|
| Head only | 0.8192 | 0 | 1.24 | 3.73 | 0.071 |
| All biases + head | **0.8810** | 0.119683 | 7.01 | 4.02 | 0.074 |
| Attention biases + head | 0.8746 | 0.042872 | 6.88 | 3.92 | 0.073 |
| MLP biases + head | 0.8722 | 0.053590 | 6.63 | 3.85 | 0.071 |
| LayerNorm biases + head | 0.8724 | 0.022329 | 6.97 | 3.90 | 0.072 |

### 2026-10-11 — Qian — configs/bitfit_336.yaml

- run_dir: `CS604_RUNS/bitfit_20261011_024724` (`last.pt`; `qq_branch_v2`).
- val acc (CodaBench): **0.734**, submission **976355**.
- trainable % / peak VRAM / GPU-hours: **0.119683%** backbone (102,912 biases; classifier excluded) / **7.01 GiB** / **2.402 wall GPU-hours**; 144 training minutes on A100 40GB. Latency: **7.403 ms/image** at batch 1, **0.641 ms/image** at batch 64, BF16 synthetic forward-only.
- what changed: final all-bias BitFit at 336 on all 450,000 training images; prototype linear head using `prototypes_336.pt`; bias/head LR 1e-3, weight decay 0, batch 64, seed 42; 5% time warmup + cosine, no epoch ceiling.

### 2026-10-10 — Wenqi — last-block attention @ 336 (`configs/lastblock_attention_336.yaml`)

- run_dir: `CS604_RUNS/last_blocks_20261010_022115`; submission file `val_team_08.txt`
- val acc (ours / CodaBench): no internal holdout / **0.679260** (macro F1 0.673014)
- trainable % / peak VRAM / GPU-hours: **2.7474%** backbone (classifier excluded) / **1.43 GiB** / **2.40** (144.01 training minutes, time limit reached)
- what changed: Qian's last-block attention recipe (CodaBench 0.668 @ 224) with `img_size: 336` and prototypes rebuilt as `prototypes_336.pt`. Same otherwise: last-block QKV + output projection, prototype linear head, AdamW lr 1e-3, weight decay 0, batch 64, 5% time warmup + cosine over 144 minutes.
- result: **+1.13 points** over Qian's 0.668 at 224, both on A100 with a 144-minute budget. Fewer epochs at 336 (3 complete epochs + 56% of epoch 4; train loss 3.20 → 1.33) than Qian's ~6 at 224. Still below Anastasia's LoRA r16 @ 336 (0.717) and full FT @ 336 (0.807). No inference latency measured.
- next: do not rerun this method. The gap is the adaptation scope, not the GPU.

### 2026-10-12 — Wenqi — eval crops 0.80 and 0.60 on full FT @ 336

- run_dir: no training. Rescored `CS604_RUNS/full_ft_20261006_122039/last.pt` with `src.predict` on `ana-branch`.
- val acc (CodaBench): crop 0.80 + flip **0.813**, submission **980303**, file `val_team_08_80.zip`; crop 0.60 + flip **0.806**, submission **980302**, file `val_team_08_60.zip`.
- trainable % / peak VRAM / GPU-hours: same checkpoint as the 2026-10-06 full FT (100% / 20.05 GB / 2.40). Prediction only, so no new training cost.
- what changed: `--crop_pct 0.80 --tta_flip` and `--crop_pct 0.60 --tta_flip` on Anastasia's full fine-tune at 336. Crop 0.75 was not rerun.
- result: crop 0.80 ties her crop 0.7 + flip score (0.813). Crop 0.60 is 0.7 points lower. Both are below Van's crop 0.75 + zoom on this same checkpoint (0.819, submission 972132).

### 2026-10-12 — Wenqi — zoom on LoRA r24 @ 336

- run_dir: no training. Rescored `CS604_RUNS/lora_20261007_051926/last.pt` with `src.predict --crop_pct 0.75 --zoom` (`ana-branch` plus `src/van_zoom.py` from `van-zoom`).
- val acc (CodaBench): **0.799**, submission **980328**, file `val_team_08_zoomcrop_075.zip`.
- trainable % / peak VRAM / GPU-hours: same checkpoint as the 2026-10-07 LoRA r24 run (4.65% of the backbone / 19.40 GB / 2.40). Prediction only.
- what changed: Van's zoom (crop 0.75, second view on the region the model used, softmax averaged). No flip.
- result: **+0.6** over this checkpoint's crop 0.7 + flip score (0.793). Same size gain Van measured on the full fine-tune (0.813 → 0.819). This is the best under-5% score so far, 0.2 points above his AdaptFormer zoom (0.797, submission 972126), and 2.0 points below the full fine-tune zoom (0.819).

### 2026-10-17 — Wenqi — zoom latency on the two submitted checkpoints

- run_dir: no training. Timed `full_ft_20261006_122039/last.pt` and `lora_20261007_051926/last.pt` with the zoom path (plain view, zoom crop, second forward). A100-SXM4-40GB, bf16, synthetic inputs, 30 warm-ups, 200 timed iterations. JSON: `inference_latency_zoom.json` in each run folder.
- full FT zoom (the 0.819 checkpoint): **20.650 ms/image** at batch 1, **1.417 ms/image** at batch 64. Single-pass latency on this checkpoint was 7.94 / 0.64 ms.
- LoRA r24 zoom, unmerged (the 0.799 checkpoint): **36.793 ms/image** at batch 1, **2.286 ms/image** at batch 64. Load printed 48 LoRA layers (qkv, proj, fc1, fc2) and 4.65% trainable. The 49 missing keys are the frozen backbone weights; a LoRA checkpoint does not store them. No unexpected keys. Single-pass unmerged latency was about 14.4–15.0 ms at batch 1; merged single-pass was 7.68 ms.


### 2026-10-14 — Qian — BitFit head, resolution and crop/flip ablations

- run_dir: `CS604_RUNS/qq_bitfit_heads_crop_336_seed42_v1/` and `CS604_RUNS/qq_bitfit_resolution_336_384_448_seed42_v1/` (`qq_bitfit_ablation`; head/crop and resolution Colab launchers, saved commits `9f1f1de` / `166df46`). Full results retained in each run root's CSVs.
- val acc (internal subset / CodaBench): at 336, linear **0.8792** vs cosine **0.8904** with standard crop; cosine crop 0.75 + flip **0.8980**. Resolution comparison at crop 0.75 + flip: 336 / 384 / 448 = **0.8976 / 0.8972 / 0.8990** / not submitted.
- trainable % / peak VRAM / GPU-hours: ~**0.12%** backbone (classifier excluded). Head ablation: **7.01 GiB**, ~**0.074** wall GPU-hours/arm. Resolution runs (336 / 384 / 448): **7.01 / 8.94 / 11.94 GiB**, **0.0745 / 0.0958 / 0.1323** wall GPU-hours, **4.03 / 5.23 / 7.30** training minutes on A100 40GB.
- what changed: compared prototype linear/cosine heads at 336, then cosine heads at 336/384/448; same 1,000 classes, 40 fit + 5 holdout images/class, fitting-only prototypes rebuilt per resolution, three epochs/arm, seed 42. Bias/head LR 1e-3, weight decay 0, batch 64, BF16, 5% warmup + cosine. Selected `best.pt` using crop 0.9 without flip; evaluated crop/flip variants on the same holdout.
- result: cosine wins every matching inference setting at 336 (+1.12 points at standard crop). Best resolution setting: **448 + crop 0.75 + flip, 0.8990** (NLL 0.418256), only +0.14 points over 336 in that comparison, with 1.81× training time. All best epochs are epoch 3; one seed, convergence unconfirmed. Subset scores do not establish full-data performance under the 144-minute budget.
- next: train BitFit with a prototype cosine head at **448** from pretrained weights on all **450,000** training images for **144 minutes**; rebuild all-data prototypes at 448, then predict with **crop 0.75 + horizontal-flip TTA** and evaluate on CodaBench.

### 2026-10-15 — Qian — BitFit cosine head @ 448

- run_dir: `CS604_RUNS/bitfit_20261015_015001/last.pt` (`qq_branch_v2`; `configs/bitfit_cosine_448.yaml`).
- val acc (CodaBench): **0.762**, submission **988896**; crop 0.75 + horizontal-flip TTA. No internal holdout.
- trainable % / peak VRAM / GPU-hours: **0.119318%** backbone (102,912 biases; classifier excluded) / **11.95 GiB** / **2.401 wall GPU-hours**; 144 training minutes on A100 40GB. Single-view latency: **7.787 ms/image** at batch 1, **1.179 ms/image** at batch 64, BF16 synthetic forward-only; excludes flip TTA and preprocessing.
- what changed: full-data BitFit at 448 with prototype cosine head (initial scale 20), all-data `prototypes_448.pt`; all 450,000 training images, bias/head LR 1e-3, weight decay 0, batch 64, seed 42, BF16, 5% time warmup + cosine. Reached epoch 6 before the time limit.
- result: **+2.8 percentage points** over the previous 336 linear-head BitFit (0.734). Resolution, head and inference settings changed together, so the gain cannot be attributed to one change.

### 2026-10-16 — Qian — last-block attention, cosine head @ 448

- run_dir: `CS604_RUNS/last_blocks_20261015_120404/last.pt` (`qq_branch_v2`; `configs/lastblock_attention_cosine_448.yaml`).
- val acc (CodaBench): **0.685**, submission **990367**; crop 0.75 + horizontal-flip TTA per launcher. No internal holdout.
- trainable % / peak VRAM / GPU-hours: **2.738970%** backbone (2,362,368 parameters; classifier excluded) / **2.23 GiB** / **2.402 wall GPU-hours**; 144 training minutes on A100 40GB. Single-view latency: **8.681 ms/image** at batch 1, **1.166 ms/image** at batch 64, BF16 synthetic forward-only; excludes flip TTA and preprocessing.
- what changed: final-block QKV + output projection only, prototype cosine head (initial scale 20), `prototypes_448.pt`, all 450,000 training images at 448; attention/head LR 1e-4 / 3e-3, weight decay 0.05 (excluding biases/norms), label smoothing 0.1, clipping 1.0, batch 64 × accumulation 4, seed 42, BF16, 5% time warmup + cosine. Reached epoch 7 before the time limit.
- result: **+0.57 percentage points** over last-block attention at 336 (0.679260), but **7.7 points below** BitFit cosine at 448 (0.762). Multiple recipe changes prevent attributing the difference to one factor; this run does not establish the method's optimized accuracy.

### 2026-10-16 — Qian — LoRA + VPT ablation (`configs/lora_vpt_ablation.yaml`)

- run_dir: `CS604_RUNS/qq_lora_vpt_336_c1000_seed42_v1/` (`qq_lora_vpt_ablation`).
- val acc (ours / CodaBench): baseline **0.9032**; shallow 8/16/32 tokens **0.9040 / 0.9044 / 0.9058**; deep 8 tokens/block **0.9072** / not submitted.
- trainable % / peak VRAM / GPU-hours: selected deep arm **4.7391%** backbone (head excluded) / **19.97 GiB** / **0.1802 wall GPU-hours** (10.03 training min).
- what changed: added VPT to LoRA r24 + extras at 336 with a prototype cosine head; 1,000 classes, 40 fit + 5 holdout/class, five epochs, seed 42; pilot-selected prompt LR **3e-4**. Deep gains **0.40 pp** over baseline with **5.4% longer training**; one seed, full-data gain unconfirmed.
- next: train **LoRA + VPT-Deep, 8 tokens/block at 336**, from pretrained weights on all **450,000 images for 144 training minutes**, using fresh all-data prototypes and prompt LR **3e-4**; evaluate on CodaBench.

### 2026-10-16 — Qian — LoRA + VPT-Deep (`configs/lora_vpt_deep_336.yaml`)

- run_dir: `CS604_RUNS/lora_vpt_deep_336_seed42/lora_20261016_050840/last.pt` (`qq_branch_v2`).
- val acc (ours / CodaBench): no internal holdout / **0.793**, submission **992462**; crop 0.75 + horizontal-flip TTA.
- trainable % / peak VRAM / GPU-hours: **4.7391%** backbone (head excluded) / **20.09 GiB** / **2.4025 wall GPU-hours**; 144 training min on A100 40GB. Merged-LoRA single-view latency: **9.430 ms/image** (batch 1), **0.694 ms/image** (batch 64), excluding flip TTA and preprocessing.
- what changed: LoRA r24 + extras with **8 VPT-Deep tokens/block at 336**, prototype cosine head using `prototypes_336.pt`, all 450,000 images; LoRA/prompt LR 3e-4, head LR 3e-3, batch 128 x 2, BF16, seed 42, 5% time warmup + cosine. Reached epoch 7.
- No improvement is demonstrated by the recorded CodaBench scores: **LoRA r24 + extras without VPT also achieved 79.3%** (crop 0.7 + flip). Different crops prevent isolating VPT's effect.

### 2026-10-18 — Qian — LoRA crop/resolution ablation (`configs/qq_lora_ablation.yaml`)

- run_dir: `CS604_RUNS/qq_lora_crop_resolution_c1000_seed42_v1/` (`qq_lora_ablation`); selected E: `E_448/lora_20261018_093724/`.
- val acc (ours / CodaBench): A/B/C/D/E final holdout **0.9082 / 0.9058 / 0.9058 / 0.9094 / 0.9122** / not submitted. Evaluation: crop 0.9, no flip or zoom.
- trainable % / peak VRAM / GPU-hours: selected E **4.9446%** backbone (head excluded) / **33.90 GiB** / **0.2183 wall GPU-hours**, excluding prototype extraction; 12 training min, 3.48 epochs on A100 40GB.
- what changed: LoRA r24 + extras, no VPT; 1,000 classes, 40 fit + 5 holdout/class, fitting-only prototypes per resolution, seed 42. Equal 12-minute arms: 336/384 with current crops or 75% crop + 25% whole-image/padding, then 448 with the mixture. E gains **+0.28 pp** over matching 384 and **+0.40 pp** over the 336 baseline. One seed; full-data gain unconfirmed.
- next: `qq_branch_v2`, `configs/lora_r24_crop_pad_448.yaml`: fresh pretrained weights, all **450,000 images**, **144 training minutes**, reuse all-data `prototypes_448.pt`; predict with **crop 0.75 + autozoom (mass 0.7), no flip** via `scripts/predict_zoom.py`.

### 2026-10-19 — Anastasia — LoRA depth controls: fixed rank on fewer blocks, `pos_embed` control, spread vs concentrated (`configs/ablation_lora_depth_control.yaml`)

- run_dir: `CS604_RUNS/ablation_ablation_lora_depth_control/` (`scripts/ablation.py`, launcher `notebooks/colab_launcher_ana_depth_control.ipynb`, `ana-branch`)
- setup: same pilot as the depth/res and param-selection screens (1,000 classes, 40 fit / 5 holdout per class, prototype heads rebuilt from fit images, 3 epochs, one seed per arm); A = `lora_r24_336_recipe.yaml`. New opt-in option `keep_embed_trainable` (with `first_trainable_block > 0`, `pos_embed` + `cls_token` still train; costs the speed-up, since backward then runs through every block). J40 and C16 rerun here so every comparison is within one session.

| Arm | Change | Holdout top-1 | vs A | Trainable | Peak VRAM | Training min |
|---|---|---:|---:|---:|---:|---:|
| A | full recipe | **0.8986** | — | 4.65% | 19.40 GB | 5.76 |
| C16 | r16, all blocks | 0.8932 | −0.54 | 3.28% | 19.34 GB | 5.78 |
| J40_embed | r40, blocks 4–11, embeddings train | 0.8926 | −0.60 | 5.06%* | 17.39 GB | 5.08 |
| J40 | r40, blocks 4–11 (rerun of depth/res J) | 0.8890 | −0.96 | 4.67% | 13.59 GB | 4.43 |
| J24_embed | r24, blocks 4–11, embeddings train | 0.8856 | −1.30 | 3.23% | 17.31 GB | 5.04 |
| J24 | r24, blocks 4–11 | 0.8830 | −1.56 | 2.84% | 13.51 GB | 4.39 |
| K24 | r24, blocks 6–11 | 0.8812 | −1.74 | 2.13% | 10.53 GB | 3.99 |

\*over the 5% cap; a control only, not a submission.

- results: **J's loss is about one third frozen embeddings, two thirds dropping the early blocks' capacity.** Training `pos_embed` + `cls_token` recovers 0.36 of J40's 0.96 points (J40_embed) and 0.26 of J24's 1.56 (J24_embed). **Spreading the budget beats concentrating it, confirmed directly:** at the same ~3.2–3.3%, r16 in all 12 blocks (C16, −0.54) beats r24 in blocks 4–11 with embeddings (J24_embed, −1.30) by **0.76 points**. Extra rank helps when the blocks are fixed (r40 over r24 in blocks 4–11: +0.60 frozen, +0.70 with embeddings). Reproducibility: J40 −0.96 here vs −0.92 in the depth/res screen; C16 −0.54 here vs C −0.66 in the r24 ablation, both within ~0.1 point.
- consistent with the earlier screens: with LoRA in every block, the extras were worth only +0.14 (r24 ablation, arm B), but once blocks 0–3 are frozen, `pos_embed` alone is worth ~0.3–0.4, so early-block LoRA appears to compensate for frozen embeddings. Together with the param-selection pilot (no depth effect for one full-rank block), the evidence favours "spread capacity across all layers" over "early layers matter".
- decision for the full-scale J run (`configs/lora_r40_blocks4_336.yaml`): **keep embeddings frozen.** J40_embed is over the 5% cap and gives up most of the speed-up (5.08 vs 4.43 training min, 17.4 vs 13.6 GB), and the full run's question is whether J's speed-up buys accuracy when epochs are limited.
- limitations: pilot scale; one seed per arm.

### 2026-10-19 — Anastasia — GPS k42 @ 336, full scale (`configs/gps_k42_336.yaml`)

- run_dir: `CS604_RUNS/selected_20261019_072529`; mask `CS604_RUNS/gps_336/gps_mask_k42.pt` (launcher `notebooks/colab_launcher_ana_fullcheck.ipynb`, `METHOD = "gps_k42_336"`, `ana-branch`)
- val acc (ours / CodaBench): no internal holdout / **0.779** plain; crop 0.7 + flip not submitted (TTA added +0.6 to +1.0 on every other 336 run)
- trainable % / peak VRAM / GPU-hours: **4.63% of the backbone** (3.98M / 85.99M, head excluded; only masked entries counted) / **20.37 GB** / 2.40 (144.0 training min, time limit reached; A100 40 GB)
- what it is: full-scale check of pilot arm G_gps_k42 (−0.52 vs LoRA A). GPS (Zhang et al., CVPR 2024): for every output neuron of every block Linear and the patch embedding, only the top 42 input connections by gradient magnitude train (mask from 400 batches = 51,200 training images, ~5/class, through the prototype head, cross-entropy); all other entries stay at their pretrained values. No LoRA. Plus the recipe's extras (norms, biases, pos_embed, cls_token). Sparse entries at `unfreeze_lr` 1e-4 (the pilot's value, not swept), extras at 3e-4; otherwise the LoRA r24 recipe (prototype cosine head, head lr 3e-3, batch 256, wd 0.05 off for these tensors, label smoothing 0.1, 5% warmup + cosine fitted to 144 min).
- training: **~8.7 epochs** (8 full + 70% of the 9th), **~993 s/epoch (~453 img/s)**: 22% faster per epoch than LoRA (1,272 s, ~6.8 epochs) and about as fast as full FT (1,017 s, ~8.5). No LoRA branches means no extra matmuls; the full weight gradients are computed and masked, so it costs full-FT-like time and memory (20.37 vs 20.05 GB full FT, 19.40 LoRA), not less. Final train loss 1.863 (full FT @336 ended at 1.87; same loss and label smoothing).
- cost: latency **7.13 ms/image at batch 1, 0.640 ms/image at batch 64** (forward only, bf16, A100). Nothing to merge: the trained entries are ordinary weights, so this is a plain ViT-B (LoRA merged 7.68–8.35 and DoRA merged 7.71 in other sessions; session-to-session variation is ~0.6 ms).
- results: **0.6 points below LoRA r24** (0.779 vs 0.785), just above Adapter (0.776) and 2.8 below full FT (0.807). **The pilot predicted this:** −0.52 on the pilot, −0.6 at full scale. So the pilot ranked GPS correctly even though GPS is the arm most like full FT, where the pilot failed (pilot −0.4 vs LoRA, full scale +2.2). Despite ~28% more epochs than LoRA (8.7 vs 6.8) and a train loss as low as full FT's, it scored lower, so its limit is what the sparse entries can express, not training time, the same conclusion as the depth/res screen. Under the 5% cap, LoRA r24 everywhere stays the best way to spend the budget.
- next: full-scale checks of B (`lora_r24_336_no_extras`) and J (`lora_r40_blocks4_336`).
