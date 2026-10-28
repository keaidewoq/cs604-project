# CS604 Presentation — Slide Outline (team 08)

Slide-by-slide outline for the 7 Nov presentation: 15 minutes + 5 minutes Q&A. Use it to
generate the slides. Numbers come from `report/final_report_outline.md` (the report
outline) and, through it, Van's draft and the progress log. If a number here disagrees
with those, fix this file.

The brief asks the slides to cover the investigation, the method's motivation, the design,
comparison with baselines and ablation studies, with figures, diagrams and charts. The
"Covers" line on each slide says which.

**How the story runs.** Ask the question on slide 2 and hold the answer back. Slides 3–5
show what we tried and the clues that it wasn't enough. Slide 6 is the turning point: we
went back to the images and found that the limit is information (fine detail between
look-alike siblings), not training time. Slides 7–9 are the fixes that follow from it, and
slide 10 sums up Task 1's answer.

**Task 2 is the next chapter, not a separate requirement.** Slide 5 showed that extra
detail only helps a model that can adapt, so slide 11 asks the natural follow-up: *is
adapting under 5% of the model enough?* Slides 12–13 answer it (where to put the 5%, and
our design using the same three ideas), and slide 14 says what training only 5% costs.
Slide 16 answers the question for both tasks.

**Wording rules:**

- Slides 3–6 are in logical order, not strict date order (the first 224 vs 336 ablation
  came before the recipe fix). Avoid "then we discovered" for the resolution clue.
- Both tasks ran in parallel. Slide 11 is a logical bridge, so don't say "after finishing
  Task 1 we moved to Task 2".

Each slide has:

- **Message:** the one sentence the audience should remember. Use it as the slide title.
- **On slide:** at most five short bullets.
- **Visual:** the figure, chart or diagram, and where its data is.
- **Say:** speaker notes.

Accuracies are CodaBench val top-1 (%) unless marked holdout.

**[TODO]** Assign a presenter to each slide. **[PENDING 30 Oct]** marks slides that wait
for the gap experiments.

---

## Timing

| # | Slide | Time | Running total |
| --- | --- | --- | --- |
| 1 | Title | 0:15 | 0:15 |
| 2 | The task and the question | 1:00 | 1:15 |
| 3 | Where we started | 1:00 | 2:15 |
| 4 | Fixing the recipe | 1:15 | 3:30 |
| 5 | The clues | 1:15 | 4:45 |
| 6 | Back to the images (turning point) | 1:15 | 6:00 |
| 7 | Fix 1: pixels, not epochs | 1:15 | 7:15 |
| 8 | Fix 2: look, then zoom in | 1:00 | 8:15 |
| 9 | Fix 3: sibling-aware kNN re-ranking | 1:15 | 9:30 |
| 10 | Our system (Task 1's answer) | 0:30 | 10:00 |
| 11 | Can 5% unlock the detail? (bridge to Task 2) | 1:00 | 11:00 |
| 12 | Where should the 5% go? | 1:00 | 12:00 |
| 13 | Task 2: our design | 1:00 | 13:00 |
| 14 | Results, and what 5% costs | 1:00 | 14:00 |
| 15 | What didn't work | 0:30 | 14:30 |
| 16 | The answer (both tasks) | 0:30 | 15:00 |
| B1–B8 | Backup slides for Q&A | | |

---

## Slide 1 — Title (0:15)

- **Message:** *Pixels, Not Epochs: Spending a Fixed Budget on Fine-Grained Recognition*
  **[TODO: agree the title with the report]**
- **On slide:** title; team 08; names; CS604, 7 Nov 2026.
- **Visual:** none, or one example image pair of look-alike species.
- **Say:** names only; go straight to slide 2.

---

## Slide 2 — The task and the question (1:00)

- **Message:** Under a fixed budget, what actually limits fine-grained accuracy?
- **Covers:** motivation.
- **On slide:**
  - 10,000 classes, only 45 training photos each; many classes are look-alike relatives
  - Budget: 96 GB-hours per training run (2 h × 48 GB)
  - Task 1: full fine-tune of ViT-B/16. Task 2: train < 5% of the backbone
  - The obvious move: make epochs cheaper so more fit (low resolution)
  - **Our question:** is training time really the limit?
- **Visual:** a 2–3 image strip of near-identical sibling classes, plus three numbers in
  large type: 10,000 classes / 45 photos / 96 GB-h.
- **Say:** set up the natural assumption ("we all thought more epochs would help") and ask
  the question. **Don't give the answer yet.**

---

## Slide 3 — Where we started (1:00)

- **Message:** Our first full fine-tune was no better than not fine-tuning at all.
- **Covers:** investigation, baselines.
- **On slide:**
  - Frozen backbone + linear probe: **62.5**
  - Full fine-tune, constant lr: **61.0**, below the frozen probe
  - First-round PEFT: last-block attention 66.8 (224 px); LoRA r16 71.7, prefix 72.1,
    BitFit 73.4 (336 px)
  - Course reference: full FT 70.8, LoRA 68.3
- **Visual:** horizontal bar chart of the five numbers, with the course references as
  dashed lines.
- **Say:** "full FT ≈ frozen probe" is the hook: something other than epoch count was
  wrong. The PEFT numbers are a first-round snapshot at mixed resolutions, not a fair
  ranking; the fair comparison comes on slide 12. Plant the Task 2 thread here: PEFT was
  stuck too, and we come back to it on slide 11.

---

## Slide 4 — Fixing the recipe (1:15)

- **Message:** Not damaging what the pretrained model already knows gained 9.4 points,
  without adding epochs.
- **Covers:** design motivation, ablation.
- **On slide** (one line each: change → effect → why):
  - Layer-wise lr decay → removing it costs **6.7** → early layers keep general features
  - Schedule decaying to zero at the time limit → 1 epoch beats 11 at a constant rate →
    weights get to settle before the clock stops training
  - Prototype head start (class means) instead of a trained probe → **+1.6 to +2.0** → a
    confident probe gives small gradients, so the backbone stops adapting
  - Head lr 10× the backbone → **+1.5** → the new head must move faster
  - Result: **66.3 → 75.7**, same 224 px, same budget
- **Visual:** waterfall chart from 66.3 to 75.7, plus the "no layer decay: −6.7" bar.
  Data: report outline Table 2.
- **Say:** the first steps came from Qian's early runs: a class-mean head start
  (61.3 → 62.1) and cosine decay (→ 66.3). Van's tuning then took it to 75.7. The
  prototype result is counter-intuitive: the probe starts 16 points higher but ends
  lower, and Anastasia reproduced it independently (−2.2 at 336, −2.9 for LoRA). The
  prototype explanation is our best hypothesis; the others are measured.

---

## Slide 5 — The clues (1:15)

- **Message:** Better training helped, but three clues said training time wasn't the limit.
- **Covers:** investigation.
- **On slide:**
  - **More epochs stopped paying:** 6 → 12 epochs only +0.5 (holdout); doubling LoRA
    epochs at equal time ≈ 0
  - **Resolution only helped once the model adapted:** frozen features got *worse* at
    higher resolution (62.5 → 62.0), BitFit (biases only) stayed flat (+0.14, Qian),
    full fine-tuning gained ~5 points (224 → 336)
  - **Augmentation and loss tweaks tied** (crop scale, colour jitter, sibling-aware
    smoothing)
- **Visual:** three small panels, one per clue: an epoch curve flattening; frozen vs
  fine-tuned bars at 224 / 336; a row of "tie" results.
- **Say:** each clue points the same way: the model is short of *information*, not of
  training time or regularisation. So we went back to the data.

---

## Slide 6 — Back to the images (turning point) (1:15)

- **Message:** The limit is fine detail: the errors are near misses between look-alike
  siblings.
- **Covers:** investigation, motivation.
- **On slide:**
  - **79%** of images have a short side of 300–400 px; training at 224 px throws detail
    away (Van)
  - Neighbouring class IDs look alike: prototype cosine **0.80** vs 0.28 for random pairs
    (Anastasia)
  - **34%** of errors fall within 5 IDs of the true class
  - Errors are near misses: the truth is 2nd in **37%** of errors, 3rd–5th in **27%**
- **Visual:** left, a sibling pair with the small deciding detail circled; right, a bar
  chart of where the true class sits among errors (2nd / 3rd–5th / beyond 5th).
  Data: report outline Section 5.3.
- **Say:** this is the reveal. **The answer to slide 2's question: information, not
  iterations.** The model needs to see small details, and it often almost has the right
  answer. The next three slides each follow from this.

---

## Slide 7 — Fix 1: pixels, not epochs (1:15)

- **Message:** At equal compute, higher resolution wins, even though fewer epochs fit.
- **Covers:** design motivation, ablation.
- **On slide:**
  - Equal-compute screen: 336 → 384 → 448 keeps gaining; 512 flat
  - Fewer epochs fit (7.7 → 4.1), and it still wins
  - The low-resolution shortcut loses: progressive 224 → 384 is **−4.3**, despite 2.5× the
    epochs
  - CodaBench, full FT: **75.7 (224) → 80.7 (336) → 81.7 (384) → 82.5 (448)**
- **Visual:** line chart, accuracy vs training resolution at equal compute, full FT and
  AdaptFormer as two lines, with epochs-that-fit labelled on each point. Data: report
  outline Table 3.
- **Say:** the screen predicted each CodaBench step. Point at the AdaptFormer line but
  don't explain it yet: "hold on to that line; it matters for Task 2" (slide 11).

---

## Slide 8 — Fix 2: look, then zoom in (1:00)

- **Message:** A second, closer look at where the model looked recovers detail.
- **Covers:** design, ablation.
- **On slide:**
  - Classify → find where the model looked (class-activation map) → crop that region from
    a 2× copy → classify again → average
  - Beats flipping at the same two passes (+0.85 / +0.62 holdout)
  - Location matters: a centred box of the same size is 0.28 worse
  - CodaBench: Task 1 **82.5 → 83.2** (crop 0.9 + zoom)
- **Visual:** one image → heat map → zoom box → crop, as a four-step strip. **[TODO, Van:
  example images]**
- **Say:** no retraining, and no extra network: it reuses the classifier's own head
  weights to find the region.

---

## Slide 9 — Fix 3: sibling-aware kNN re-ranking (1:15)

- **Message:** The right answer is often in the top 5, so re-rank the top 5 by the
  closest training photos.
- **Covers:** design motivation, ablation.
- **On slide:**
  - Why: the truth is in the top 5 for **64%** of errors, many of them look-alike siblings
  - How: score = log p + 20 × (mean of the 3 closest training photos of that class)
  - Individual photos beat class averages (+1.3 vs +0.1 to +0.7)
  - Settings fixed once on the holdout, never tuned on CodaBench, same for every model
  - CodaBench: **+1.40** (LoRA plain), **+0.85** (Task 1 → **84.07**), **+0.88** (Task 2
    → **83.08**)
- **Visual:** diagram: photo → top-5 candidates → each compared with that class's 3
  closest training photos → re-ordered list. Small table of the three CodaBench gains.
- **Say:** this follows directly from slide 6: the error analysis showed the answer was
  usually nearby, so re-order the candidates instead of re-classifying. Costs no training
  and < 0.1 ms per image; the cost is a ~0.7 GB feature bank. (Anastasia's analysis and
  design; Wenqi ran it on Task 2.)

---

## Slide 10 — Our system (0:30)

- **Message:** Task 1's answer in one picture: keep the pretrained features, train on more
  pixels, look closer at the near misses.
- **Covers:** design (framework diagram).
- **On slide:** the framework diagram, labelled with the gain of each stage.
- **Visual:** report outline Figure 1, drawn properly: image → 448 px → ViT-B/16 → cosine
  head → zoom pass → average → top-5 kNN re-rank → prediction. **[TODO: draw once, reuse in
  the report]**
- **Say:** recap of slides 4–9 in one picture: 61.0 → 84.07 with every weight trainable.
  End on the question for the next slide: *but what if you can only change 5% of it?*

---

## Slide 11 — Can 5% unlock the detail? (1:00)

- **Message:** Extra detail only helps a model that can adapt. Is adapting under 5% of it
  enough?
- **Covers:** motivation (for Task 2), investigation.
- **On slide:**
  - From slide 5: the gain from resolution depends on how much of the model adapts
  - Frozen: worse at higher resolution · biases only (BitFit, 0.12%): flat, +0.14 ·
    LoRA (4.9%): +0.40 · AdaptFormer (4.1%): **+1.55 per step**, not flat at 448
  - So Task 2 asks two questions: *is 5% enough?* and *where should the 5% go?*
- **Visual:** one bar per method, "gain from higher resolution", ordered by how much of the
  model trains. Data: report outline Section 5.2 table.
- **Say:** this is why Task 2 belongs in the same story: it tests our answer under a harder
  constraint. Be clear that the bars come from different experiments (Van's equal-compute
  screen, Qian's BitFit and LoRA pilots), so it's a pattern, not one controlled curve.
  The first answer is yes: AdaptFormer gains even more per step than full fine-tuning.

---

## Slide 12 — Where should the 5% go? (1:00)

- **Message:** **[PENDING 30 Oct]** pick the matching version:
  - if the spread claim holds: *Spread the budget across every layer.*
  - if it doesn't: *The size of the budget matters more than where it goes.*
- **Covers:** comparison with baselines, ablation.
- **On slide:**
  - Fair comparison: one resolution (336), one recipe, one view, equal training minutes,
    10,000-class holdout
  - Methods: LoRA, AdaptFormer, serial adapter, BitFit, last-block attention, prefix;
    plus LoRA concentrated in blocks 4–11
  - LoRA vs AdaptFormer: 3 seeds each, so we can say whether the difference is real
  - Already pointing the same way (Qian): prefixes in all 12 blocks beat the last block by
    6.5 points; a 10× longer prefix adds nothing
- **Visual:** bar chart of holdout accuracy per method, coloured by "all layers" vs
  "concentrated / small", with backbone % under each bar and error bars for the 3-seed
  arms. Data: `ablation_ablation_peft_controlled/arms/*.json`.
- **Say:** explain why we re-ran the survey: the first round mixed resolutions, recipes
  and views (slide 3), so it couldn't rank methods fairly. Qian's ablations back it up:
  all biases beat any one type of bias, and QKV in the last two blocks is worse than in
  one.
- **Ready now:** if the 30 Oct results are late, this slide can run on Qian's prefix
  placement chart alone: last block 65.1 → last 4 blocks 69.0 → all blocks 71.7.

---

## Slide 13 — Task 2: our design (1:00)

- **Message:** Yes, 5% is nearly enough: 83.08% with 4.94% of the backbone, using the same
  three ideas as Task 1.
- **Covers:** design, ablation.
- **On slide:**
  - AdaptFormer branch beside every MLP (starts as the identity), LoRA r16 on attention,
    all biases and LayerNorms train
  - Step-by-step gains (holdout): design tweaks +0.99, biases/norms +0.50, LoRA +0.57
  - Same three ideas: same recipe pieces · 448 px (77.9 → 80.4 → 81.2 single view) · zoom
    + kNN with unchanged settings (→ **83.08**)
  - Why this combination: **[PENDING 30 Oct, Van's equal-parameter controls]**
- **Visual:** left, the one-block diagram (report outline Figure 2) with trainable parts
  highlighted; right, a step chart of the ablation chain.
- **Say:** the point for the story is that nothing new was needed at prediction time:
  zoom and kNN transferred from Task 1 untouched. If the equal-parameter controls show the
  combination matters, say AdaptFormer and LoRA are complementary (MLP path vs attention).
  If not, say the gain came from extra parameters, and that the main design gains are
  biases/norms and 448 px.

---

## Slide 14 — Results, and what 5% costs (1:00)

- **Message:** Both tasks beat the course references by more than 13 points; training only
  5% costs 1 point and saves memory, not time.
- **Covers:** comparison with baselines; all four metrics; trade-offs.
- **On slide:** one table:

  | | Course ref. | Task 1 full FT | Task 2 PEFT |
  | --- | --- | --- | --- |
  | Top-1 (CodaBench val) | 70.8 / 68.3 | **84.07** | **83.08** (−1.0) |
  | Trainable (backbone) | 100% / 0.34% | 100% | **4.94%** |
  | Training | | 2.4 h A100, 96 GB-h, 17.8 GB peak | same time, **14.7 GB** peak |
  | Latency, batch 1, with zoom | | 23.4 ms | 29.9 ms (LoRA merged) |
  | Storage | | 375 MB model | 17 MB adapters + 31 MB head |

- **Visual:** the table with the better value in each row highlighted.
- **Say:** both used exactly the 96 GB-h budget, with the schedule ending at the time limit.
  PEFT saves memory, not time: gradients still flow through the frozen backbone. The kNN
  bank adds ~0.7 GB to either pipeline, so the storage saving is for the model, not the
  full pipeline. The brief expected PEFT to beat full fine-tuning on small data; at 450,000
  images it didn't.

---

## Slide 15 — What didn't work (0:30)

- **Message:** Changing the loss or training longer didn't help; adding information did.
- **Covers:** investigation, ablation.
- **On slide:** two columns.
  - Didn't help: progressive low resolution, longer training at 224, LP-FT,
    sibling-aware losses and margins, heavier augmentation
  - Didn't help for PEFT: DoRA (tie, 25% slower), AdaLoRA (−2.7), VPT (no full-scale
    gain), longer prefixes, SSF, concentrating the budget in a few layers
- **Visual:** the two-column list, crossed-out style.
- **Say:** mention the excluded class-count balancing in one line (+1.7, but it uses the
  known per-class counts of val and test, so we don't report it as a result).

---

## Slide 16 — The answer (0:30)

- **Message:** Under a fixed budget, the limit was information, not iterations, even when
  only 5% of the model can change.
- **Covers:** motivation, conclusion.
- **On slide:**
  - Keep what the pretrained model knows (recipe: +9.4)
  - Spend compute on pixels, not epochs (448 px: +6.8)
  - Look closer at the near misses (zoom + kNN: +1.6)
  - Task 1: 61.0 → **84.07**. Task 2: **83.08** with 4.94% of the backbone, same three ideas
- **Visual:** the CodaBench progression chart (Figure 3), both tasks, with the three stages
  shaded.
- **Say:** close by answering slide 2's question in one sentence, for both tasks. Then
  "Thank you, questions".

---

## Backup slides (Q&A only)

- **B1. How we evaluated without spending submissions.** Labelled holdout (5 training
  photos per class held out, never trained on), seed noise ~0.3, and the 1,000-class
  pilots vs the 10,000-class holdout.
- **B2. Task 1 ablation table.** Report outline Table 2.
- **B3. First-round PEFT survey.** Report outline Table A1, labelled as uncontrolled.
- **B4. kNN details.** Re-ranker comparison; settings chosen on one half of the holdout and
  confirmed on the other; fixes vs breaks about 2 : 1; the bank holds no trained
  parameters **[TODO: the Prof's answer on whether it counts toward Task 2's 5%]**.
- **B5. Latency details.** Protocol, LoRA merging is exact (100% agreement), zoom costs
  2.5–2.8× a single pass at batch 1.
- **B6. Loss curves.** Both finals; why Task 2's loss is higher (CosFace margin × learned
  temperature).
- **B7. Class-count balancing.** +1.7 (82.8 → 84.5), and why it's excluded.
- **B8. Per-method PEFT ablations (Qian).** Report outline Table 5b: last-block scope,
  prefix placement and length, BitFit scope, head and resolution, VPT, LoRA at 448,
  LoRA vs DoRA vs AdaLoRA.

---

## Figures to make

Each figure is reused in the report where possible.

| Figure | Slides | Data | Owner |
| --- | --- | --- | --- |
| Sibling example pairs | 1, 2, 6 | dataset images | **[TODO]** |
| First-results bar chart | 3 | report outline Section 4.2 | **[TODO]** |
| Recipe waterfall | 4 | Table 2 | **[TODO]** |
| Three-clue panels | 5 | Tables 2, 4; Section 5.2 | **[TODO]** |
| Where the truth sits among errors | 6 | Section 5.3 | Anastasia |
| Resolution at equal compute | 7 | Table 3 | **[TODO]** |
| Zoom example strip | 8 | Van's zoom outputs | Van |
| kNN re-ranking diagram | 9 | Section 3.6 | Anastasia |
| Framework diagram | 10 | Figure 1 | **[TODO]** |
| Gain from resolution vs how much adapts | 11 | report outline Section 5.2 | **[TODO]** |
| Controlled PEFT bar chart | 12 | 30 Oct results | **[PENDING]** |
| Prefix placement chart (fallback for 12) | 12 | report outline Table 5b | Qian |
| Task 2 block diagram + ablation steps | 13 | Figure 2, Table 6 | Van |
| Results and cost table | 14 | Table 1 | **[TODO]** |
| CodaBench progression chart, both tasks | 16 | Figure 3 | **[TODO]** |
