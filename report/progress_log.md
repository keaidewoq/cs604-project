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
