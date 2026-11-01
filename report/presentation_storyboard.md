# Accuracy–Efficiency Trade-offs in Adapting Pretrained Vision Transformers for Fine-Grained Classification

**Presentation storyboard | 19 main slides + appendix | 7 November 2026, 12:00 (15 min + 5 min Q&A)**

Revised 2026-10-29 from the first storyboard, checked against `report/progress_log.md`
(`ana-branch`), Van's report draft (`report/final_report_draft.md` on `van-dev`, 2026-10-28),
Van's progress-log entries and `van_full_ft_findings.md` / `van_zoom_crop_findings.md`
(`van-448-finals`). CodaBench scores are written as percentages (0.825 → 82.5).

> **Narrative rule:** Organise by research questions, **not chronology**. Explain resolution as
> an independently tested accuracy–compute trade-off. Every plotted comparison must identify
> subset/full-scale, holdout/CodaBench, resolution, inference view (plain / flip / zoom / kNN)
> and whether the recipes are matched.
>
> **Open team decision:** Van's draft (§2.3) motivates resolution by fine-grained detail (79% of
> images have a short side of 300–400 px; the deciding cue is often small). This storyboard
> treats resolution as a tested trade-off and mentions the image-size statistic only as a data
> property. Agree on one framing before building slides.

**Timing:** 19 slides ≈ 47 s each. If over time: merge slides 10 and 11, and move slide 18 to
the appendix.

## Holdouts and evaluation settings (use these labels on every chart)

| Label | What it is | Who |
| --- | --- | --- |
| **CodaBench** | public val, 50,000 images, 5/class; headline metric | all |
| **Van holdout** | all 10,000 classes, 40 fit / 5 held out per class, split seed 604; laptop (RTX 5080) screens | Van |
| **Van equal-compute screen** | Van holdout, but 10 training images/class, equal GB-h per arm | Van |
| **All-class holdout** | all 10,000 classes, 40 fit / 5 held out, split seed 0; A100, 144 min | Anastasia |
| **Qian all-class** | all 10,000 classes, 40 / 5, seed 42 (selected-attention screen) | Qian |
| **1,000-class pilot** | 1,000 random classes, 40 / 5, seed 0 (Ana) or 42 (Qian); 3–5 epochs | Anastasia, Qian |
| **Class-block pilot** | 100 blocks of 10 consecutive class IDs (siblings present) | Anastasia |
| **~2,000-class resolution subset** | class count inferred from timings; 29 min per arm | Anastasia |

Pilot accuracies (~88–91%) are inflated by the easier task: use them for direction and
ranking only. Seed noise ≈ 0.3 pp; treat differences under ~0.4 pp as ties.

---

## Main deck

### Slide 1 — Title and scope
**Headline:** Accuracy–Efficiency Trade-offs in Adapting Pretrained Vision Transformers for Fine-Grained Classification.
- 10,000 species, 45 training photos each; timm `vit_base_patch16_224.augreg2_in21k_ft_in1k`.
- Team 08, CS604, 7 Nov 2026, member names (fill in).

**Visual:** minimal title with 2–3 genuine images from the project dataset; no graph.

### Slide 2 — Why is this task challenging?
**Message:** Fine-grained recognition requires separating similar species despite variable appearance within a species.
- Low inter-class differences: neighbouring class IDs are visually close (prototype cosine 0.80 at 1 ID apart vs 0.28 for random pairs).
- High intra-class variation: part, life stage, season, scale.
- Only 45 photos per class; ImageNet-pretrained features are useful but need adaptation.

**Visual REQUIRED:** two image pairs from verified project examples: look-alike siblings (e.g. close-ID pairs from the error gallery) and one species in two looks (e.g. adult moth vs caterpillar, class 1020). Label as illustrative.

### Slide 3 — Research questions, two tasks and constraints
**Message:** We investigate *how to adapt* under a fixed compute budget and a parameter cap.
- RQ1 full FT: how to optimise under fixed compute?
- RQ2 PEFT: where and how to train < 5% of the backbone?
- RQ3 extension: what errors remain, and can prediction-time methods help?
- **Budget:** 2 h × 48 GB = 96 GB-h. Every final ran on one A100 40 GB for 144.0 min = 2.4 h × 40 GB = **96 GB-h**, stopped by the clock; the cosine schedule ends exactly at the limit.
- **Rules (Prof, 2026-10-03):** the head is always trained and not counted in the 5%; the 5% applies to the backbone.
- Outside the training clock, reported separately: class prototypes (one frozen pass), AdaptFormer PCA calibration (~10 s), kNN feature bank (~15 min at 336, ~25–30 min at 448). **Confirm with the Prof** that the bank is reported beside the budget.

**Diagram REQUIRED:** AugReg2 backbone → Task 1 full FT (100%) and Task 2 PEFT (< 5%) → shared prediction pipeline (zoom, kNN) and error analysis. No chronological arrows between the two task branches.

### Slide 4 — Experimental design and fair comparison
**Message:** Controlled ablations are kept apart from full-scale leaderboard results.
- 450k training images, 10k classes; several holdouts (table above).
- Full-scale runs: all 45 photos/class, 144 min, CodaBench only (no labels).
- Prediction views: plain crop 0.9; crop + flip; zoom; zoom + kNN. Class-count balancing excluded (team decision 2026-10-27).
- Metrics: Top-1 (pp differences), trainable backbone %, peak VRAM, GPU-hours, latency.

**Visual REQUIRED:** protocol matrix: rows = holdout types; columns = classes / images per class / seed / used for. Legend: controlled vs confounded comparison.

### Slide 5 — Baselines and starting point
**Message:** Our first full fine-tune barely beat a frozen probe, so something other than epoch count was wrong.

| Reference | Top-1 | Trainable | Source |
| --- | ---: | ---: | --- |
| PhD reference: zero-shot, initialised head | 43.7 | 0% | course brief |
| PhD reference: full FT | 70.8 | 100% | course brief |
| PhD reference: LoRA | 68.3 | 0.34% | course brief |
| Ours: prototype head, no training | 43.9 (Van holdout) | 0% | Van |
| Ours: frozen probe, 224 / 336 | 62.5 / 62.0 | 0% | Anastasia |
| Ours: first full FT, constant lr, 120 / 144 min | 61.0 / 61.3 | 100% | Qian |
| + class-mean head init | 62.1 | 100% | Qian |
| + cosine decay (shared trainer) | 66.3 | 100% | Qian |
| Last-block attention, 224 | 66.8 | 2.75% | Qian |

**GRAPH REQUIRED:** horizontal bars, PhD references in grey, ours in colour, CodaBench vs holdout marked. This sets up the finals on slide 16 (84.07 / 83.08).

### Slide 6 — Task 1: the full-FT recipe
**Message:** Optimisation choices, not more epochs, moved full FT from 66.3 to 75.7 at the same 224 px and budget (+9.4 pp).

Van holdout, 3 epochs, one change at a time:

| Change | Result | Effect |
| --- | --- | --- |
| Control (LP-FT start, head shares lr 4e-4, layer decay 0.75) | 69.83 | — |
| No layer decay | 63.12 | **−6.7** |
| Head lr 10× (4e-3) | 71.37 | +1.5 |
| Prototype head start instead of LP-FT | 72.98 vs 71.37 / 71.03 | **+1.6 to +2.0** |
| Crop scale [0.35, 1] / no colour jitter / head lr 30× | ties | — |
| Training length 3 / 6 / 12 epochs | 72.98 / 73.97 / 74.49 | diminishing |

- Warmup + cosine decayed to zero at the time limit: one epoch with it beats 11 epochs at a constant rate.
- Head-start finding replicated: LP-FT −2.2 at 336 (Anastasia's full-FT pilot) and −2.9 for LoRA.
- Final recipe: prototype cosine head, layer-wise lr decay 0.75, top lr 4e-4, head lr 4e-3, 5% warmup + cosine fitted to 144 min, label smoothing 0.1, drop_path 0.1, clip 1.0, batch 256, bf16.
- CodaBench at 224: 66.3 → **75.7** (plain) → 77.3 (crop 0.7 + flip).

**GRAPH REQUIRED:** layer-decay bars (63.1 vs 69.8) plus head-start bars (71.37 vs 72.98), with compact recipe table. Label "Van holdout, 3 epochs, one seed (seed spread 0.33 pp)".

### Slide 7 — Task 1: does higher resolution justify fewer epochs?
**Message:** At equal compute, higher resolution beat more epochs, up to 448 px; low-resolution shortcuts lost every time.
- ~2,000-class subset, 29 min per arm (AugReg2): 224 81.03% (19 epochs); 336 **84.72%** (9); progressive 224→336 82.87% (18). DINOv2 control: 83.29 / 86.09 / 85.24 (not allowed for submission).
- 336 leads 224 at equal minutes from ~6 min on by 3.5–3.9 pp; both plateau, so 336 has a higher ceiling.
- Van equal-compute screen (10 img/class, crop 0.9): full FT 336 / 384 / 448 / 512 = 63.20 / 64.06 / 64.79 / 64.98 (epochs 7.7 / 5.5 / 4.1 / 2.9; flat at 512). AdaptFormer 336 / 384 / 448 = 59.63 / 61.18 / 62.72 (not yet flat).
- Progressive 224 → 384 at equal compute: **−4.3** (Van). LoRA 224 → 336 pilot: −1.20 / −0.88 at equal epochs / time.
- **Resolution helps only when the backbone adapts:** frozen prototype accuracy falls 43.9 / 42.5 / 37.2 at 224 / 336 / 448, and the frozen probe goes 62.5 → 62.0, while fine-tuned models gain ~5 pp from 224 to 336.
- CodaBench, full FT plain: 224 75.7 → 336 80.7 → 384 81.7 → **448 82.5**. Caveat: the 384 and 448 runs used Van's `_fast` configs (torch.compile, 12 workers, batch 64 × 4 at 448), so they are not perfectly matched to the 336 run.

**GRAPH REQUIRED (primary):** accuracy vs elapsed minutes for 224 / 336 / progressive from `history_subset_<backbone>_<arm>.csv`, vertical line at 29 min. **GRAPH REQUIRED (secondary):** Van's equal-compute curve (accuracy vs training size, both methods) with the CodaBench plain points overlaid. Do not fabricate curves from summary numbers.

### Slide 8 — Task 2: PEFT method design
**Message:** Different PEFT methods spend the parameter budget in different places.
- Selected attention (last block), prefix, BitFit, serial adapter, AdaptFormer, LoRA, DoRA, GPS; VPT and AdaLoRA as extensions.
- Trainable backbone %: BitFit 0.12; prefix 1.07; last-block attention 2.75; LoRA r24 4.12 (4.65 with extras); AdaptFormer 4.13; GPS 4.63; adapter 4.68; DoRA 4.75; **final AdaptFormer + LoRA r16 + biases/norms 4.94**.

**DIAGRAM REQUIRED:** side-by-side simplified ViT block, frozen weights grey, trained parts highlighted, % labels. Include the final Task 2 block: LayerNorms and all biases trainable; LoRA r16 on qkv; parallel AdaptFormer branch beside the frozen MLP (down 768→192, PCA init, ReLU, up 192→768 zero init, × learned scale). No accuracy ranking on this slide.

### Slide 9 — Task 2: which methods balance accuracy and parameters?
**Message:** Parameter count alone does not decide accuracy; methods that touch every block cluster at the top.

CodaBench, 336 px, plain crop 0.9, 144 min:

| Method | Backbone % | Top-1 | Peak VRAM | Latency, batch 1 | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| Last-block attention | 2.75 | 67.93 | 1.43 GiB | not measured at 336 | linear head, lr 1e-3 |
| Prefix, length 50 | 1.07 | 72.1 | 7.97 GiB | 9.64 ms | linear head |
| BitFit (all biases) | 0.12 | 73.4 | 7.01 GiB | 7.40 ms | linear head |
| Serial adapter w96 + extras | 4.68 | 77.6 | 15.54 GB | 11.60 ms | not mergeable |
| GPS k42 | 4.63 | 77.9 | 20.37 GB | 7.13 ms | plain ViT at inference |
| AdaptFormer w192 | 4.13 | 77.9 | TBC (laptop 7.6 GB at batch 64) | not mergeable | Van's recipe, CosFace 0.1 |
| DoRA r24 + extras | 4.75 | 78.3 | 28.06 GB | 23–24 ms unmerged / 7.71 merged | |
| LoRA r24, no extras | 4.12 | 78.4 | 19.30 GB | 15.37 ms unmerged / 7.78 merged | |
| LoRA r24 + extras | 4.65 | 78.5 | 19.40 GB | 14.2–15.0 ms unmerged / 7.68–8.35 merged | |

- Context: an early LoRA r16 with a weak head recipe scored 71.7; the shared recipe added ~6.8 pp.
- 77.6–78.5 is within ~1 pp, close to seed noise. Methods differ in head type, learning rate and epochs reached, so this is an observed pipeline ranking, not a controlled causal one.
- **Placeholder:** controlled comparison on the all-class holdout (`configs/ablation_peft_controlled.yaml`: same head, 20 min, plain crop, with seeds for LoRA and AdaptFormer), due 2026-10-30. Replace or annotate this slide's ranking with it when it lands.

**GRAPH REQUIRED:** scatter, x = trainable backbone % (log scale), y = Top-1, one label per method, 336 plain only. **TABLE REQUIRED:** the table above. Latencies come from different sessions (batch-1 variation ~1 ms).

### Slide 10 — Task 2 ablation: how should LoRA capacity be allocated?
**Message:** Spread the adaptation across all blocks rather than concentrating a similar budget.
- 1,000-class pilot, depth controls: r16 in all 12 blocks, 3.28% → 89.32%; r24 in blocks 4–11 with trainable embeddings, 3.23% → 88.56% (**+0.76 pp** to spreading). r24 in all blocks, 4.65% → 89.86%.
- About one third of the loss from freezing blocks 0–3 is the frozen embeddings; the rest is lost early-block capacity.
- Parameter-selection pilot: one full-rank attention block costs ~1 pp at any depth (S0 ≈ S5 ≈ S11), so depth itself does not matter; the most concentrated arm (4 single matrices) loses 2.58; GPS (every layer) −0.52.

**DIAGRAM + GRAPH REQUIRED:** 12-block strips showing trained layers, next to a 3-bar holdout plot with %. Label "1,000-class pilot, one seed".

### Slide 11 — Task 2 ablation: classifier-head optimisation
**Message:** Head initialisation and head learning rate can outweigh smaller PEFT changes.
- 1,000-class pilot, LoRA r24: prototype cosine, fast lr 3e-3 → 89.84%; prototype cosine, slow lr 1e-4 → 83.70%; probe head fast → 86.92%; probe head slow → 85.72%; prototype linear fast → 88.92%.
- The prototype start wins only with a fast head lr (+2.9 fast, −2.0 slow). Of the 2.9 pp gap to LP-FT, ~2.0 is the prototype start and ~0.9 the cosine head.
- Same direction elsewhere: full FT prototype vs LP-FT +1.6 to +2.0 (Van); BitFit cosine vs linear head +1.12 (Qian, pilot).
- Why the head is trained at all: a frozen-head LoRA underfit (61.6% on its own training images; 10× the budget added only +1.1).

**GRAPH REQUIRED:** 2×2 interaction lines (init × head lr) with actual scores; separate marker for the prototype linear head. No unqualified causal claim about gradients.

### Slide 12 — Task 2 final design: AdaptFormer + LoRA, step by step
**Message:** Each kept change was screened on the holdout and confirmed on CodaBench.

Van holdout, 336 px, one change at a time:

| Step | Change | Holdout | Gain | Backbone % |
| --- | --- | ---: | ---: | ---: |
| 1 epoch | adapter lr 1e-3 (3e-4 / 1e-3 / 3e-3 / 1e-2 = 70.56 / 71.49 / 71.49 / 70.45) | 71.49 | | 4.13 |
| 1 epoch | CosFace margin 0.1 | 72.09 | +0.60 | |
| 1 epoch | adapter reads norm2(x), not the residual | 71.72 | +0.24 | |
| 1 epoch | learned per-block scale | 71.63 | +0.15 | |
| 1 epoch | PCA init of the down projection | 71.56 | +0.08 | |
| 3 epochs | all four combined vs control | 76.26 vs 75.27 | +0.99 | 4.13 |
| 3 epochs | + trainable biases and LayerNorms | 76.76 | +0.50 | 4.27 |
| 3 epochs | + LoRA r16 on qkv | 77.33 | +0.57 | 4.96 (4.94 at 448) |

CodaBench confirmations:

| Run | Plain | Best view |
| --- | ---: | --- |
| AdaptFormer, 336 | 77.9 | 79.7 (zoom) |
| + biases/norms, 336 | — | 80.2 (zoom) |
| same at 384 | 79.4 | 80.7 (zoom) |
| same at 448 | 80.4 | 81.4 (crop 0.9 + zoom) |
| **+ LoRA r16, 448** | **81.2** | **82.2** (crop 0.9 + zoom) |

- Caveats on the slide: AdaptFormer and LoRA tie at 336 (77.9 vs 78.4–78.5), so do not claim AdaptFormer is better in itself; the gains came from biases/norms, LoRA and 448. AF + LoRA (4.94%) vs AF + biases (4.27%) may partly be the extra 0.7% of parameters; Van's equal-parameter controls (AF width 224; LoRA r28 + biases/norms) are pending.

**GRAPH REQUIRED:** waterfall of holdout gains, then a small CodaBench line (336 → 384 → 448 → + LoRA).

### Slide 13 — What limits predictions after adaptation?
**Message:** Remaining errors are mostly look-alike relatives and unfamiliar appearances.
- All-class holdout, LoRA r24 no extras: Top-1/2/3/5/10 = 77.8 / 86.0 / 89.2 / 92.0 / 94.4%; 34% of errors within 5 class IDs. Van's full-FT holdout: 46% within ±10 IDs.
- The ID proxy holds up: each class's most similar class lies within 5 IDs for 50% of classes (chance 0.10%).
- Siblings are the main difficulty once present: the same recipe scores 83.5 on the class-block pilot vs 89.8 on random classes, with 60% of errors on relatives (18% on random classes).
- Pilot error flags (overlapping, heuristic): unseen look 59%, sibling 47%, messy class 45%, two looks 10%. "Unseen look" uses the same model's features, so its 5.9× over-representation is an upper bound.
- LoRA and full FT make different errors (pilot): ~30% of each one's errors are right under the other; 71% are shared.

**GRAPH REQUIRED:** top-k curve and error-flag bars (59 / 47 / 45 / 10). **IMAGE GALLERY REQUIRED:** 4–6 verified mistakes from `CS604_RUNS/error_analysis/gallery_*.png` and `error_causes/causes_<cause>.png` (sibling pair, life stage/season, tiny subject).

### Slide 14 — Prediction-time extension 1: look, then zoom in
**Message:** A second, closer view of where the model found its evidence adds accuracy without retraining, and beats flip for the same two passes.

**How it works (Van):** classify; the patch tokens most similar to the predicted class's head weight mark the evidence (a class activation map); the smallest square holding 70% of that map is cut from a 2× copy of the image, resized to the input size, classified again, and the two softmaxes are averaged.
- **Beats flip at the same two passes** (Van holdout, eval crop 0.75): AdaptFormer 77.95 vs 77.10; full FT 336 79.59 vs 78.97. Adding both flips to zoom (4 passes) adds only ~0.05–0.13.
- **Where it zooms matters:** a centred box of the same size scores 77.67 / 79.43, so most of the gain is the close-up and the rest is aiming it. The aimed zoom helps most on small subjects (+2.5 vs +0.9 on the 691 holdout images with the tightest boxes, full FT).
- **CodaBench:** full FT 336 81.3 (crop 0.7 + flip) → 81.9 (crop 0.75 + zoom); AdaptFormer 336 79.0 → 79.7; AdaptFormer 384 plain 79.4 → 80.7 zoom (+1.3).
- **Retuned at 448:** the tighter 0.75 crop removes context the zoom box needs once the image is already large, so crop 0.9 + zoom won (chosen on half of Van's holdout, +0.28 on the other half): full FT 82.8 → 83.2, AdaptFormer 80.9 → 81.4 on CodaBench.
- **Zoom gains shrink as resolution rises** (full FT plain → zoom: +1.2 at 336, +0.7 at 384, +0.7 at 448 with crop 0.9): resolution and zoom partly overlap.
- **Cost:** 2.2–2.8× single-pass latency (box search and the 2× resize add fixed overhead at batch 1): full FT 448 8.47 → 23.43 ms at batch 1; zoom at 448 builds 896 px copies and needs a lot of host memory (check the row count before submitting).

**DIAGRAM REQUIRED:** image → pass 1 → class activation map → 70%-mass box → crop from 2× copy → pass 2 → average softmax. Show 2–3 real zoom boxes (one small subject). **GRAPH REQUIRED:** paired bars plain / + flip / + zoom for full FT and AdaptFormer at 336, with the centred-box control as a third bar; label "Van holdout, 2 passes each".

### Slide 15 — Prediction-time extension 2: kNN re-ranking
**Message:** Matching a photo to individual training photos fixes near misses that the classifier head misses, with no training and the same settings on every model.

**How it works (Anastasia):** take the model's top 5 classes; for each, find the 3 training photos of that class closest to the query (cosine on the model's features) and re-score with `log p + 20 × mean(top-3 cosine)`.
- **Why top 5:** on the all-class holdout about two thirds of errors have the truth in the top 5 (2nd in 37%, 3rd–5th in 27%), so re-ordering a short list targets them.
- **Settings fixed before any CodaBench file:** K = 5, m = 3, β = 20 chosen on one half of the all-class holdout classes and scored on the other, then swapped; both halves picked the same values and gained +1.33 / +1.25 (fixes vs breaks ≈ 2 : 1).
- **Individual photos beat class averages** (bakeoff, plain view, all-class holdout): NCM +0.55; 3 k-means centres +0.70; kNN max +1.18; **kNN mean3 +1.32**; mean over all photos +0.13. A learned linear re-ranker adds only +0.2 over kNN mean3 once zoom is used: not worth the complexity.

| Model | Before | After kNN | Gain |
| --- | ---: | ---: | ---: |
| LoRA r24 no extras, 336, plain | 78.42 | 79.82 | +1.40 (same pass) |
| Full FT 448, crop 0.9 + zoom | 83.22 | 84.07 | +0.85 (same pass) |
| AdaptFormer + LoRA 448, crop 0.9 + zoom | 82.2 | 83.08 | +0.88 (different passes, same checkpoint and settings) |

- Gains shrink on stronger models and on top of zoom (+1.40 → +0.85 / +0.88): zoom already fixes some of the same near misses.
- **Cost:** no training and no trainable parameters; a 450k × 768 fp16 feature bank ≈ 0.7 GB, built once outside the training clock (~15 min at 336, ~25–30 min at 448); re-ranking 0.03–0.06 ms per image, negligible next to the forward pass. The bank is a lookup memory, not trained parameters (confirm reporting with the Prof).
- Class-count balancing (+1.7, 82.8 → 84.5) is excluded: it uses the known 5/10 images per class of val/test, which a deployed model would not know.

**DIAGRAM REQUIRED:** query (after zoom) → model top 5 → 3 nearest training features per candidate → `log p + 20 × mean(top-3 cosine)` → prediction. **GRAPH REQUIRED:** bakeoff method-gain bars (all-class holdout) plus the three CodaBench before/after pairs, with "same pass" / "different passes" marked.

### Slide 16 — Final systems at 448 px
**Message:** With the same budget and the same prediction pipeline, the PEFT model stays within 1.0 pp of full fine-tuning while training 4.94% of the backbone.

| | Task 1: full FT 448 | Task 2: AdaptFormer w192 + LoRA r16 (qkv) + biases/LayerNorms, 448 |
| --- | --- | --- |
| CodaBench, plain crop 0.9 | 82.5 (ID 995914) | 81.2 (ID 1008585) |
| + crop 0.9 + zoom | 83.2 (ID 1004167) | 82.2 (ID 1008590) |
| + zoom + kNN (K = 5, m = 3, β = 20) | **84.07** | **83.08** |
| Trainable backbone | 100% (93.93M incl. head) | **4.94%** (4,262,412 / 86.25M; 11.94M incl. head) |
| Peak VRAM | 17.80 GB | 14.66 GB |
| Training | A100 40 GB, 144 min, 2.40 h = 96 GB-h, 5.5 epochs | same, 5.5 epochs |
| Latency, batch 1 / 64, one pass | 8.47 / 1.17 ms | 13.45 / 1.56 ms (11.27 / 1.33 merged) |
| Latency, batch 1 / 64, with zoom | 23.43 / 2.54 ms | 33.78 / 3.32 ms (29.88 / 2.86 merged) |
| kNN extra | ~0.7 GB bank, < 0.1 ms/image | ~0.7 GB bank, 0.062 ms/image (batch 64) |
| Run folder | `full_ft_20261017_122237` | `van_adaptformer_20261021_083823` |

| View | Task 1 | Task 2 | Gap |
| --- | ---: | ---: | ---: |
| Plain | 82.5 | 81.2 | 1.3 |
| Crop 0.9 + zoom | 83.2 | 82.2 | 1.0 |
| Zoom + kNN | 84.07 | 83.08 | 1.0 |

**GRAPH REQUIRED:** grouped bars, x = plain / zoom / zoom + kNN, two bars per group, gain labels in pp. Optional inset: CodaBench progression of both tasks (Task 1: 61.0 → 66.3 → 75.7 → 80.7 → 81.7 → 82.5 → 83.2 → 84.07; Task 2: 66.8 → 71.7 → 78.5 → 80.2 → 81.4 → 82.2 → 83.08). The progression mixes plain and zoom views: label each point's view and the change behind each jump.
**Caveats on the slide:** latencies from one A100 session (2026-10-28); no class-count balancing; kNN settings not tuned on CodaBench; Task 2's +0.88 compares two passes (see open items).

### Slide 17 — Task 1 vs Task 2: the accuracy–efficiency trade-off
**Message:** Full FT keeps an accuracy edge at this data scale; PEFT restricts trainable parameters and memory, but not training time.
- Controlled-ish comparison at 336 plain: full FT 80.7% / 100% / 20.05 GB; LoRA extras 78.5% / 4.65% / 19.40 GB; LoRA no extras 78.4% / 4.12% / 19.30 GB; adapter 77.6% / 4.68% / 15.54 GB; DoRA 78.3% / 4.75% / 28.06 GB.
- The gap shrank from 2.8 pp (336, early October: 80.7 vs 77.9) to 1.0 pp at the finals.
- Training time is not saved: both finals reached 5.5 epochs. Per-epoch time at 336: adapter 951 s, GPS 993, full FT 1,017, LoRA 1,272, DoRA 1,585.
- Storage per new task: 17 MB of adapters + 31 MB head vs a 375 MB model.
- Inference: LoRA and DoRA merge exactly (merged LoRA r24 at 336: 7.78 ms vs 7.78 ms for a plain ViT-B, same session); AdaptFormer and adapters cannot be merged (+33% batch-1 latency for the merged Task 2 final vs full FT).
- **Professor's few-shot expectation, partial evidence:** on the 1,000-class pilot (40k images) LoRA matched or beat full FT (full FT −0.4), while at full scale full FT won by 2.2. At 10 images/class on all 10,000 classes, full FT still won (64.79 vs 62.72 at 448). So any crossover looks tied to total dataset size, not images per class. A proper few-shot sweep was not run.

| If you care most about | Choose |
| --- | --- |
| Best accuracy | Full FT 448 + zoom + kNN (84.07) |
| Best accuracy under 5% | AdaptFormer + LoRA 448 + zoom + kNN (83.08) |
| Lowest latency under 5% | LoRA r24 merged (78.5 plain / 79.9 zoom at 336, same speed as plain ViT-B) |
| Fewest parameters | BitFit 0.12% (76.2 at 448, crop + flip) |

**GRAPH REQUIRED:** (A) Top-1 vs peak VRAM, (B) Top-1 vs trainable %; 336 plain runs plus the two 448 finals as distinct markers. **SMALL TABLE:** merged/unmerged latencies with session and batch size.

### Slide 18 — What didn't work
**Message:** Many plausible ideas tied or lost; the investigation narrowed down what matters.

| Idea | Result | Setting |
| --- | --- | --- |
| Progressive low-res training (224 → 336 / 384) | −1.85 (full FT subset); −4.3 (Van screen); −1.2 / −0.88 (LoRA pilot) | equal compute |
| Longer training at 224 | 6 → 12 epochs only +0.5 | Van holdout |
| LP-FT head start | −1.6 to −2.9 | full FT and LoRA |
| Taxonomy-aware label smoothing | ties; 0.2 smoothing −0.86 | class-block pilot |
| Margin against siblings | −1.3 to −7.3 (errors scatter to far classes) | class-block pilot |
| Bigger / aimed training crops | ties (−0.28); box finder no better than random | pilot |
| Heavier augmentation (RandAugment, adapter dropout) | ties | Van holdout |
| SSF instead of biases/norms | tie (+0.21), 21% slower | Van holdout |
| DoRA | tie, +45% VRAM, 25% slower | full scale |
| AdaLoRA | −2.7 vs LoRA | 448 pilot |
| VPT-Deep on LoRA | +0.40 pilot; no full-scale gain shown | |
| Freezing early blocks to speed up | −0.9 to −1.6 | pilot |
| Learned linear re-ranker | +0.2 over kNN mean3 with zoom | all-class holdout |
| Class-count balancing | works (+1.7) but excluded | uses known class counts |

**VISUAL:** two-column "kept / dropped" table; colour by direction of effect.

### Slide 19 — Conclusions, limitations and future evaluation
**Message:** Effective adaptation is a joint problem of resolution, optimisation, parameter allocation and prediction design.
- **RQ1:** the recipe (+9.4 pp) and resolution 224 → 448 (+6.8 pp, CodaBench plain) mattered most; higher resolution beat more epochs at equal compute. Final 84.07.
- **RQ2:** spread capacity across all blocks, train the head fast from prototypes; parameter-efficient ≠ compute-efficient. Final AdaptFormer + LoRA 83.08 at 4.94%, 1.0 pp behind full FT.
- **RQ3:** errors are mostly siblings and unfamiliar looks; loss reshaping did not help, while zoom and kNN (+0.85 to +1.40) did, at extra inference cost and a 0.7 GB bank.
- Limitations: pilot vs full-scale mismatch; one-seed ablations; recipe confounds in the method survey; no catastrophic-forgetting measurement; ID-distance taxonomy proxy; no few-shot sweep.

**VISUAL REQUIRED:** three columns (RQ1–3), each with one verified metric and one caveat. Don't duplicate slide 17's frontier.

---

## Appendix (prepare; show only if asked)

- **A1:** complete experiment index: run ID, holdout, seed, resolution, recipe, trainable %, time, VRAM, Top-1, view, latency (`report/experiment_index.csv`).
- **A2:** prefix length 5 / 20 / 50: 71.648 / 71.634 / 71.688% (pilot); placement last block / last 4 / all.
- **A3:** AdaLoRA at 448 (pilot): LoRA r24 90.76%; AdaLoRA t8 / t16 / t20 87.92 / 87.94 / 88.04%; equal 12 min, unequal epochs (2.07 vs 3.42).
- **A4:** DoRA vs LoRA at 336: 78.3 vs 78.5 plain; VRAM 28.06 vs 19.40 GB; ~5.45 vs ~6.8 epochs. Qian at 448 pilot: DoRA r23 −0.28.
- **A5:** BitFit 336 vs 448: 73.4 vs 76.2, but head, resolution and TTA all changed. BitFit resolution pilot flat (336 / 384 / 448 = 89.76 / 89.72 / 89.90).
- **A6:** test crop and flip (FixRes effect): crop 0.7 + flip +1.6 at 224 (75.7 → 77.3), +0.6 at 336, crop 0.9 best at 448; zoom tuning sweep at 448 (crop × mass × min box, chosen on half the holdout).
- **A7:** training crops and box finding: DDT and prototype boxes no better than random placement; centred crops beat them; aimed crops untested for lack of a localiser.
- **A8:** taxonomy smoothing and sibling margin tables (class-block pilot).
- **A9:** compute accounting: 96 GB-h per final; work outside the clock; tuning compute (Van laptop ~17.5 h; pilots 4–12 min each). Van's speed check: loader and GPU step both ~450–500 img/s; torch.compile + 12 workers +24–33%.
- **A10:** latency protocol (A100, bf16, synthetic inputs, 30 warm-ups + 200 iterations) and LoRA merge validation (fp32 max logit diff 1.5e-5 / 8e-6, 100% top-1 agreement); session variation ~1 ms at batch 1.
- **A11:** full error galleries; per-ID-range difficulty; rank-2 errors spread across a third of classes (no short list of problem pairs).
- **A12:** kNN bakeoff full table (plain / zoom, with and without balancing) and learned re-ranker weights.
- **A13:** backbone screen: frozen prototypes 41.8 (orig in21k) / 43.9 (augreg2) / 50.0 (augreg in21k) / 53.8 (DINOv2); linear probe augreg2 59.0 ≈ DINOv2 58.6 > DeiT III 52.6 > CLIP 41.7. Backbone fixed by the course.
- **A14:** Qian's selected-attention screen (Qian all-class holdout): classifier only 62.70; last-block QKV 64.91; two-block QKV 64.06; last-block full attention 65.41.
- **A15:** LoRA speed-up arms: blocks 4–11 / 6–11, progressive 224 → 336, at equal epochs and equal minutes; all −0.9 to −1.6.
- **A16:** VPT on LoRA: shallow 8 / 16 / 32 and deep 8 tokens (pilot); full-scale 79.3 crop + flip, same as LoRA without VPT.

## Required source data and plotting checklist

| Slide | Plot or visual | Source | Priority |
| --- | --- | --- | --- |
| 2 | Species image pairs | error galleries, verified | Required |
| 3 | Task 1 / Task 2 branches + shared pipeline | course brief, configs | Required |
| 4 | Holdout protocol matrix | holdout table above | Required |
| 5 | Baseline bars | course brief; Qian / Ana / Van entries | Required |
| 6 | Layer-decay + head-start bars | `van_full_ft_findings.md` §4.3, §4.5 | Required |
| 7 | Accuracy vs elapsed minutes | `history_subset_<backbone>_<arm>.csv` | Required |
| 7 | Equal-compute resolution curve | Van's 2026-10-16 / 10-17 / 10-18 entries | Required |
| 8 | PEFT block diagrams incl. final Task 2 block | configs; Van draft §3 | Required |
| 9 | Accuracy vs backbone % (336 plain) | full-scale run entries | Required |
| 10 | LoRA depth strips + bars | 2026-10-19 depth controls; 2026-10-17 param selection | Required |
| 11 | Head init × lr interaction | 2026-10-12 LoRA recipe ablation | Required |
| 12 | AdaptFormer waterfall + CodaBench line | Van draft §2.6; Van's entries | Required |
| 13 | Top-k + error flags + gallery | 2026-10-26 holdout; 2026-10-25 causes | Required |
| 14 | Zoom diagram + example boxes; zoom vs flip vs centred-box bars | Van zoom entries (2026-10-09, 10-19, 10-20); `van_zoom_crop_findings.md` | Required |
| 15 | kNN diagram; bakeoff gain bars; before/after pairs | 2026-10-26 holdout, 2026-10-27 bakeoff and submissions, 2026-10-28 Wenqi | Required |
| 16 | Final systems grouped bars | Van draft §0, §5.3; 2026-10-27 / 10-28 kNN entries | Required |
| 17 | Top-1 vs VRAM / trainable % | run entries; Van draft §5.3 | Required |
| 18 | Kept / dropped table | entries listed in the table | Required |
| 19 | Findings summary | synthesis of controlled evidence | Required |
| Report | Loss curves of both finals | `CS604_RUNS/<run>/figures/`, `epoch_log.csv`; Van draft §5.5 | Required (report) |

## Additional evaluation (status as of 2026-10-29)

1. **Paired error comparison** (fixed / broken / shared) between models on the same holdout: **script ready** (`scripts/paired_compare.py`). Do not imply resolution causality unless recipes match.
2. **McNemar + bootstrap intervals** (by photo and by class) for small differences: covered by the same script. Image-level uncertainty does not measure seed variation.
3. **Controlled PEFT comparison** on the all-class holdout (`configs/ablation_peft_controlled.yaml`), with seeds for LoRA and AdaptFormer: ready to run, due 2026-10-30.
4. **Equal-parameter controls at 4.94%** (AdaptFormer width 224 + biases/norms; LoRA r28 + biases/norms): Van, in his holdout setup.
5. **Per-class recall and confusion:** macro recall, class-level distribution, look-alike examples.
6. **End-to-end latency and memory** under one protocol: done for the two finals (Van, 2026-10-28); other models are from earlier sessions.
7. ~~Matched 336 / 384 / 448 ablation~~: **done** (Van's equal-compute screen, flat at 512).
8. ~~Few-shot sweep~~: **cut** (does not fit by 2026-10-30; the pilot builder cannot cap images per class). State as a limitation.
9. Human-reviewed appearance annotations, forgetting assessment, SupCon / multi-prototype: future work.

## Open items before building slides

- **Task 2 same-pass zoom file:** the 82.2 zoom score is Van's earlier submission (ID 1008590); the zoom-only file from Wenqi's kNN pass was never submitted. Either footnote the +0.88, or submit that file before the leaderboard closes (30 Oct 23:59) to make it a same-pass comparison.
- **Narrative framing:** settle research-question order vs Van's problem-driven order, and whether the image-size statistic motivates resolution (see the note at the top).
- **Prof check:** kNN feature bank reported beside the 96 GB-h budget, not inside it.
- **AdaptFormer 336 VRAM** on the A100 (only the laptop's 7.6 GB at batch 64 is recorded).
- **Van's draft §3** says Task 2 stops at the averaged softmax; out of date now that Task 2 uses kNN (83.08).
- **Phase 2 test files (31 Oct – 6 Nov, 2 submissions):** Task 1 84.07 pipeline and Task 2 83.08 pipeline.

## Presentation quality-control rules

- One main conclusion per slide, supported by a figure or a clear comparison.
- Put **pilot vs full scale**, **holdout vs CodaBench**, **plain vs flip / zoom / kNN**, and **one-seed caveats** directly on the affected chart.
- Use **percentage points (pp)** for differences; never mix with relative-percent gains.
- Never say all PEFT runs were optimised equally, or that PEFT must outperform full FT.
- Never claim a 0.1–0.4 pp difference is significant without a paired test.
- Compare latencies only within one session and at the same batch size.
- Use the original CSVs for curves and exact scores; never reconstruct data points.
- Balanced (class-count) numbers appear only as an excluded analysis.
