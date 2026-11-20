# Experiment record

Folders beside this file were renamed so the name states the config. This file still uses the original short ids. The "Original id to folder" table in `README.md` (section "Van's laptop runs") maps each id to its folder.

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

## Full FT learning-rate schedule ablation: constant vs cosine to zero, 3 epochs (2026-11-11)

Tracker row F13 and `report/RERUNS.md` item 4. Does decaying the LR to zero matter, with nothing
else changed? `full_proto_constlr` is `full_proto_headlr10x` with only the new key
`lr_schedule: constant`: the same 5% linear warmup (234 of 4,689 steps), then the rate holds
at its peak (4e-4 top block, layer decay 0.75; head 4e-3) to the end. Recipe otherwise
identical: ViT-B/16 augreg2_in21k_ft_in1k, 224 px, prototype cosine head, batch 64 x 4, wd 0.05,
label smoothing 0.1, drop_path 0.1, clip 1.0, bf16, seed 0; Van holdout (all 10k classes,
40 fit / 5 held out, split seed 604, 50,000 images, plain view). Step-zero gate 43.900 (passed).

| Run | Schedule | ep1 / ep2 / ep3 | Holdout loss ep3 |
| --- | --- | --- | --- |
| `full_proto_headlr10x` (reference) | warmup + cosine to zero | 60.580 / 70.172 / **72.984** | 1.259 |
| `full_proto_constlr` | warmup + constant | 58.168 / 62.352 / 63.990 | 1.657 |

- Difference: -2.41 / -7.82 / **-8.99** per epoch. A clear loss (tie rule 0.4); one seed.
- Losses match the reference step for step through warmup; afterwards the constant arm's
  training loss plateaus near 3.1-3.3 (reference reaches ~2.43 in epoch 3).
- Parameters: 93,478,657 total, all trainable (backbone 85,798,656, 100%; head 7,680,001).
- RTX 5080 Laptop at full power (175 W limit), peak 5.82 GB, 4,191.9 s, 1.164 GPU-hours,
  0.386 normalized 48-GB hours. No predictions or CodaBench submission.
- Commit `f142505` with the uncommitted `lr_schedule` change (`git_dirty: true`):
  `src/cs604/engine/budget.py`, `src/cs604/engine/loop.py`, `tests/test_budget.py`, config.
- Output: `C:\Users\Van Le\cs604-project\cs604-outputs\full_proto_constlr\` (`best.pt` = epoch 3).
- Conclusion: decay to zero is worth ~9 points at our peak LR in 3 epochs. A lower constant rate
  was not tested. Replaces the uncontrolled "1 epoch beat 11 constant epochs" comparison.

## Task 2 equal-parameter controls for AdaptFormer + LoRA r16, 3 epochs (2026-11-03/04)

Is the AF + LoRA r16 gain (+0.57, entry "AdaptFormer + biases/norms + LoRA" below) just more
parameters? Each control matches its 4.96% budget, with either wider adapters alone or LoRA alone on every
linear layer. New key `lora_targets` (default `[qkv]`) and `adapter_dim: 0` (no adapters).
Configs are `configs/experiments/eqp_{af_lora16,af224,lora28,af_bn}_e3_s{0,1}.yaml`, run at commit 2b82c3e
(clean) from the Colab notebook. The recipe is the same as `af_lora_r16_e3`: 336 px; Van holdout (all 10k classes, 40 fit
/ 5 held out per class, split seed 604, 50,000 images, plain view); 3 epochs (4,689 steps); batch
64 x 4; lr 1e-3; head lr 4e-3; CosFace 0.1; label smoothing 0.1; wd 0.05; bf16. Every run passed the step-zero gate
at 42.536. `_s0` and `_s1` differ only in the training seed.

| Run | Backbone trainable | Seed 0 ep1 / ep2 / ep3 | Seed 1 ep1 / ep2 / ep3 | Mean ep3 | Diff vs AF+LoRA (s0 / s1 / mean) | Holdout loss s0 / s1 |
| --- | --- | --- | --- | --- | --- | --- |
| `eqp_af_lora16_e3`: AF w192 + biases/norms + LoRA r16 qkv | 4,262,412 (4.96%) | 69.23 / 75.55 / **77.44** | 69.08 / 75.54 / **77.41** | **77.43** | | 1.156 / 1.162 |
| `eqp_af224_e3`: AF w224 + biases/norms | 4,262,796 (4.96%) | 69.48 / 75.48 / 76.81 | 69.43 / 75.18 / 76.87 | 76.84 | -0.63 / -0.55 / -0.59 | 1.194 / 1.195 |
| `eqp_lora28_e3`: LoRA r28 qkv/proj/fc1/fc2 + biases/norms, no AF | 4,250,880 (4.94%) | 64.21 / 74.28 / 77.23 | 64.41 / 74.10 / 77.28 | 77.26 | -0.21 / -0.13 / -0.17 | **1.134 / 1.135** |
| `eqp_af_bn_e3`: AF w192 + biases/norms (reference) | 3,672,588 (4.27%) | 69.44 / 75.43 / 76.77 | n/a / n/a / 76.77 | 76.77 | -0.67 / -0.65 / -0.66 | 1.195 / n/a |

- **Two-seed reading (final):** at equal parameters, the combination beats wider adapters by 0.59
  (0.63 and 0.55: a win in both seeds by the 0.4 rule). Width 224 adds nothing over 192 (+0.07). The
  combination **ties LoRA alone** at equal parameters (+0.17; 0.21 and 0.13). LoRA alone has the lowest
  holdout loss in both seeds, even though it starts slowest (64.2/64.4 at epoch 1). So the slide-12
  claim is "adding LoRA to AdaptFormer beats spending the same parameters on wider adapters (+0.6, two
  seeds)"; "the combination beats a pure-LoRA design" is **not** supported (tie). Seeds agree within
  0.08 per arm.
- **Seed 1 (2026-11-04)** ran at commit 2b82c3e on a Colab G4 (RTX PRO 6000 Blackwell, 96 GB; seed 0
  was on the A100 40 GB). Step-zero accuracy 42.51 (seed 0: 42.536), within the ±0.2 check. Median
  training img/s on the G4: AF + LoRA 601, AF w224 675, LoRA r28 497. Drive filled up during the
  queue: `eqp_af_bn_e3_s1` lost its `metrics.csv`/`train.log` (score from `run.json`, status
  completed, `best_internal_accuracy` 0.76766, hence n/a per epoch), and `eqp_af_lora16_e3_s1` and
  `eqp_af224_e3_s1` lost a top-level `last.pt` or a final hash-log line. Their final checkpoints were
  checked against the numbered step-4689 copies and copied by hand (`sync.json` says "salvage copy").
- **Reproducibility:** the two arms that repeat earlier laptop runs agree with them. AF + biases/norms gives
  76.77 vs 76.756 (`af_bn_bias_norm_e3`), and AF + LoRA r16 gives 77.44 vs 77.330 (`af_lora_r16_e3`). With the same
  recipe on a different GPU and session, the noise is about 0.1, well under the 0.4 rule.
- **Speed** (median training img/s on the A100 40 GB, for the two runs that stayed on one GPU): AF + LoRA
  391.8, AF w224 441.0 (11% faster). The other two runs were split across an A100 and an L4, so
  their speeds are not comparable.
- **Run notes:** session 1 dropped at step ~3150 of `eqp_lora28_e3_s0` (Colab disconnect). The run
  resumed from `resume_step_003126.pt` on an L4 (~80 img/s), which Colab assigned in place of an A100.
  The 3 h watchdog then stopped `eqp_af_bn_e3_s0` at step ~2150, and it resumed from
  `resume_step_002000.pt` on an A100. A resume reloads the saved optimizer, scheduler and batch
  position, so these count as uninterrupted runs. Check `nvidia-smi` before starting a
  queue.
- All 8 runs synced to `cs604-outputs/eqp_*` (2026-11-04) and their numbered checkpoints pruned
  from Drive.

## Slide 10 spread vs concentrated, LoRA and AdaptFormer, 2 seeds (2026-11-04/05)

RERUNS item 3. Same recipe and holdout as the equal-parameter controls above. The concentrated
arms put the same parameters in blocks 4-11 only (0-based, the last 8 of 12), moved rather than
added; biases and LayerNorms still train in every block. LoRA: rank 42 on qkv/proj/fc1/fc2,
4,250,880 (4.94%), vs `eqp_lora28_e3` (rank 28 in all 12 blocks, same count). AdaptFormer:
width 288 in blocks 4-11 (`adapter_dim: [0]*4 + [288]*8`), 3,669,512 (4.27%), vs `eqp_af_bn_e3`
(width 192 in all 12, 3,672,588). New key `lora_blocks`. LoRA s0 at commit 2274398, the other
three at b1c87cd; all on a Colab G4 except `eqp_af288_b4_e3_s0`, which ran on an L4 to step
~4,450, was stopped by the 3 h watchdog and resumed from `resume_step_004000.pt` on a G4.
Step-zero 42.51-42.52 in all four.

| Run | Seed 0 ep1 / ep2 / ep3 | Seed 1 ep1 / ep2 / ep3 | Mean ep3 | Spread reference mean | Concentrated minus spread (s0 / s1 / mean) | Holdout loss s0 / s1 (spread) |
| --- | --- | --- | --- | --- | --- | --- |
| `eqp_lora42_b4_e3`: LoRA r42, blocks 4-11 | 64.11 / 74.42 / 77.33 | 64.54 / 74.29 / 77.32 | **77.33** | 77.26 (77.23 / 77.28) | +0.10 / +0.04 / **+0.07** | 1.124 / 1.127 (1.134 / 1.135) |
| `eqp_af288_b4_e3`: AF w288, blocks 4-11 | 69.51 / 75.42 / 76.97 | 69.44 / 75.28 / 76.83 | **76.90** | 76.77 (76.77 / 76.77) | +0.20 / +0.07 / **+0.13** | 1.192 / 1.194 (1.195 / n/a) |

- **Reading: a tie for both methods.** Moving the whole LoRA or adapter budget out of blocks 0-3
  costs nothing: concentrated is nominally ahead in all four seed pairs, but every gap is under
  0.4 (largest +0.20). The holdout loss is also equal or slightly lower when concentrated. So
  "spread the adaptation across all blocks" is **not** supported on all 10,000 classes, for either
  method. This agrees with the team's controlled 20-minute comparison (concentrated LoRA r40 vs
  spread r24, -0.22, tie) and not with the 1,000-class pilot (+0.76 for spreading, one seed).
  The pilot's concentrated arm also froze the biases and norms of blocks 0-3 (it did train the
  embeddings), so its loss is more likely from freezing those than from where the LoRA sat;
  here they train in every block in both arms. That is an explanation, not a tested one.
- **Limits:** one split (40/5, seed 604), 3 epochs at 336, ~4-5% budget, one concentration
  pattern (blocks 4-11). Extreme concentration still hurts: last-block attention alone scored
  67.9 on CodaBench (slide 9).
- **Cost:** concentrating is slightly cheaper. Peak memory LoRA 9.15 vs 10.07-10.13 GB, AF 7.84 vs
  8.16 GB. On the G4, LoRA r42 in blocks 4-11 trained at 564 img/s vs 497 for LoRA r28 in all
  blocks (seed 1, +13%); AF w288 at 714 img/s (the AF w192 seed-1 log was lost, so no same-GPU
  pair for AdaptFormer).
- All four synced to `cs604-outputs` and pruned from Drive (2026-11-05).
- `eqp_lora42_b4_e3_s0`: median 563 img/s on the G4. Drive filled up at the end of the run: training and evaluation
  finished (the score is in `metrics.csv` and `train.log`), but `run.json` was never marked completed and
  `last.pt` on Drive stayed at step 4500. The local copy takes `last.pt` from
  `resume_step_004689.pt` (hash not logged) and `best.pt` from the matching `best_step_004689.pt`.

## Flipped copies added to the zoom + kNN pipeline (2026-10-30)

Evaluation only, `scripts/eval_rerank_beta.py --flip`, same holdout, banks and protocol as the
beta re-check below. Variants: `zoom3` = softmax mean of plain + flip + zoom, `zoom4` = also the
flipped zoom (as cs604-main `src.predict --zoom --tta_flip`); `+query_pf` = kNN query
normalize(plain + flipped plain feature). Change vs the current submission (2-pass zoom, plain
query, K 5, m 3, beta 20), points (paired z), photo by photo:

| Run | zoom4, b15 | **zoom4, b20** | zoom4, b25 | zoom4 + query_pf, b20 | zoom3, b20 | zoom + query_pf, b20 |
| --- | --- | --- | --- | --- | --- | --- |
| `af_lora_r16_e3` (336) | +0.13 (2.6) | **+0.10 (2.1)** | +0.06 (1.2) | +0.10 (2.0) | −0.18 (−3.4) | −0.00 |
| `full_proto_336_len3` (336) | +0.16 (3.2) | **+0.08 (1.7)** | +0.05 (1.0) | +0.11 (2.2) | −0.12 (−2.3) | +0.07 |
| `res_screen_448` (448) | −0.11 (−1.9) | **+0.05 (1.1)** | +0.04 (0.8) | +0.06 (1.1) | −0.08 (−1.6) | +0.02 |
| `af_res_screen_448` (448) | −0.00 (0.0) | **+0.12 (2.8)** | +0.13 (2.6) | +0.11 (2.1) | −0.04 (−0.7) | +0.04 |

- **4 passes + kNN beta 20 wins on all four, +0.05 to +0.12.** Beta stays 20 (336 runs lean to
  15, 448 screens to 20-25). 3 passes lose: plain counts twice against zoom once. The flipped
  query adds nothing reliable.
- **CodaBench (val), both finals re-predicted on the laptop with Anastasia's cached banks
  (cs604-main `scripts/van_predict_rerank_flip.py`, van-dev bf8f8f4): Task 1 0.8415 vs 0.8407
  (+0.08), Task 2 0.8317 vs 0.8308 (+0.09)**, as the holdout predicted. Laptop 2-pass outputs
  differ from the Colab files on 122 / 168 of 50,000 re-ranked predictions (bf16 near-ties).
- Cost: twice the forward passes of the zoom path; ~16 min per model for val on the laptop.

## kNN re-rank weight beta re-checked for the zoom submissions (2026-10-30)

Evaluation only, `scripts/eval_rerank_beta.py`, full internal holdout (50,000), laptop RTX 5080
at 175 W, `last.pt` of each run. It reproduces the team's submission re-rank (cs604-main
`scripts/predict_rerank.py`, Anastasia): crop 0.9 (model default) plain view, zoom probabilities
`(softmax plain + softmax cam m0.7 zoom) / 2`, top K re-scored by `log p + beta * knn`, knn =
mean of the m best cosines between the plain-view feature and the class's bank photos. The bank
is the run's own training photos (eval view, crop 0.9), as on CodaBench. Settings chosen on one
half of the holdout classes and scored on the other, then swapped. The question: the team uses
K 5, m 3, beta 20, fixed on Anastasia's LoRA 336 holdout without zoom (grid 10 / 20 / 40);
do the 448 finals with zoom want another beta?

Gain over no re-rank (points), K 5, m 3, **zoom view** (the submitted path):

| Run (holdout proxy) | Size | Bank | Zoom top-1 | b10 | b15 | **b20** | b25 | b30 | b40 | Chosen on halves |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `res_screen_448` (full FT, 10/class) | 448 | own 100k | 65.724 | +1.41 | +1.70 | **+1.88** | +1.84 | +1.83 | +1.69 | 20 / 25 |
| `af_res_screen_448` (AdaptFormer, 10/class) | 448 | own 100k | 63.814 | +1.49 | +1.77 | **+1.90** | +1.95 | +1.97 | +1.81 | 25 / 30 |
| `af_lora_r16_e3` (Task 2 recipe, 40/class) | 336 | own 400k | 78.880 | +0.71 | +0.80 | **+0.86** | +0.86 | +0.78 | +0.63 | 25 / 20 |
| `full_proto_336_len3` (full FT, 40/class) | 336 | own 400k | 79.706 | +0.75 | +0.84 | **+0.81** | +0.77 | +0.66 | +0.46 | 15 / 15 |

- **Beta 20 stays.** With zoom the optimum is a flat 15-30 on all four models, and beta 20 is
  within 0.07 points of each model's best. Beta 40 is 0.1-0.35 worse; no other value is a clear win.
- The plain view prefers a higher beta (25-40 on the 336 runs; e.g. AdaptFormer + LoRA b40
  +1.43 vs b20 +1.19): zoom already fixes some of the same near misses. Not used, since both
  finals submit zoom.
- K 10 instead of 5, or m 1 / 5 instead of 3, changes the held-out gain by at most 0.1.
- A first pass gave the two 10/class screens a 40/class bank: the kNN then saw 4x the model's
  training photos and beta ran to the grid edge (80, +6.5 to +7.3). That mismatch is not the
  CodaBench setting, so those rows (`analysis/rerank_beta_bank40pc.json`) are discarded.
- Cost on the laptop: 448 bank 100k ~8 min, 400k ~30 min; holdout plain + zoom ~4 min.
- Outputs: `<run>/analysis/rerank_beta.json`, stores in `analysis/rerank_beta/`.

## Task 2 AdaptFormer + biases/norms + LoRA, 1 epoch (2026-10-20)

Does LoRA on qkv add to the Task 2 recipe? New key `adapter_lora: true` (`mode: adaptformer`):
every qkv wrapped in LoRA rank 8 / alpha 16 (294,912), beside the width-192 adapters and the
trainable biases + LN weights (the wrapped `qkv.base.bias` still trains). `af_lora_e1` =
`af_bn_bias_norm_e1` + `adapter_lora`, LoRA at the adapter lr 1e-3. Commit ccac542 (clean),
336 px, 40/5 split, 1 epoch (1,563 steps), laptop at 175 W, keep-awake held.

| Run | Beside adapters + head | Backbone trainable | Step 0 | Holdout top-1 | Diff | Holdout loss | Train img/s | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `af_bn_bias_norm_e1` (control) | biases + LN weights | 3,672,588 (4.27%) | 42.534 | 72.966 | | 1.207 | 186.7 | 8.19 GB |
| `af_ssf_e1` | SSF | 3,756,300 (4.37%) | 42.534 | 73.178 | +0.21 | 1.195 | 147.6 | 14.64 GB |
| `af_lora_e1` | biases + LN weights + LoRA r8 qkv | 3,967,500 (4.61%) | 42.534 | **73.544** | **+0.58** | 1.180 | 174.7 | 8.69 GB |
| `af_lora_r16_e1` | biases + LN weights + LoRA r16 qkv (alpha 32) | 4,262,412 (4.96%) | 42.534 | **73.744** | **+0.78** | 1.173 | 176.1 | 8.70 GB |

- **A win** by the 0.4 threshold (+0.58 = 289 images), and consistent: holdout loss -0.027,
  training loss lower at every checked step (250: 6.366 vs 6.444; 500: 5.913 vs 5.951;
  1000: 5.330 vs 5.385; 1550: 5.099 vs 5.157).
- **Cost: 6% lower throughput** (174.7 vs 186.7 img/s), +0.5 GB. Under a time-capped final run
  that is ~6% fewer steps, much less than SSF's 21%. 1 epoch only: the gap may shrink with
  more epochs (as it did for other early-training gains), so a 3-epoch check
  (vs `af_bn_bias_norm_e3` 76.756) is the next step before a Colab final.
- **Rank 16 vs rank 8: +0.20 (a tie by 0.4)**, at no throughput cost (176.1 vs 174.7 img/s,
  noise) and the same memory; holdout loss -0.007; training loss mixed (250: 6.366 vs 6.366;
  500: 5.935 vs 5.913; 1550: 5.102 vs 5.099). Rank 16 is the largest that fits under 5%
  (rank 17 is 5.00%), so it leaves no room for anything else. Commit 095d04f.
- **3 epochs (2026-10-21, `af_lora_r16_e3`, commit 2f93683): 77.330 vs 76.756 (+0.57, a win)**,
  so the 1-epoch gain holds. Per epoch vs `af_bn_bias_norm_e3`: 69.274 / 75.658 / 77.330 vs
  69.616 / 75.430 / 76.756 (-0.34 / +0.23 / +0.57; the gap opens late, with the lr decay).
  Holdout loss 1.159 vs 1.195; training loss at step 4650 4.487 vs 4.562. Cost: 169.1 vs
  186.6 img/s (9% slower over 3 epochs), same 8.7 GB. Under a time-capped final, about 9%
  fewer steps; at 3 epochs the control gains +1.33 from epoch 2 to 3, so 9% fewer steps
  should cost well under the +0.57. Ported to the team trainer (van-dev 1022022, same logits).
- **CodaBench (2026-10-21)**: run `CS604_RUNS/van_adaptformer_20261021_083823`
  (`configs/adaptformer_w192_lora-r16-qkv_448px_bias-and-layernorm_cosface0.1_2.4h.yaml`), A100 40 GB, 144 min, 5.5 epochs (6.1 without
  LoRA), 14.66 GB peak, all 45/class. **0.812 plain** (ID 1008585) and **0.822 crop 0.9 + zoom**
  (ID 1008590) vs 0.804 / 0.814 without LoRA: **+0.8 on both**, more than the screen's +0.57
  despite 10% fewer epochs. Task 2 best. Latency 12.4 ms at batch 1 on the A100.

## Task 2 AdaptFormer + SSF in place of biases/norms, 1 epoch (2026-10-20)

Does SSF beside the adapters beat training the pretrained biases and LayerNorm weights? New key
`adapter_ssf: true` (`mode: adaptformer`): the SSF scale/shift after every op (205,824) next to
the width-192 adapters; with `adapter_input: norm` the adapter reads the scaled norm2 output.
`af_ssf_e1` = `af_bn_bias_norm_e1` with `adapter_train_bias`/`adapter_train_norm` off and
`adapter_ssf` on, SSF at the adapter lr 1e-3. Commit dff3140 (clean), 336 px, 40/5 split,
1 epoch (1,563 steps), laptop at 175 W.

| Run | Beside adapters + head | Backbone trainable | Step 0 | Holdout top-1 | Diff | Holdout loss | Train img/s | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `af_bn_control_e1` | nothing | 3,550,476 (4.13%) | 42.534 | 72.396 | -0.57 | 1.235 | 207.6 | 7.64 GB |
| `af_bn_bias_norm_e1` (control) | biases + LN weights | 3,672,588 (4.27%) | 42.534 | 72.966 | | 1.207 | 186.7 | 8.19 GB |
| `af_ssf_e1` | SSF | 3,756,300 (4.37%) | 42.534 | **73.178** | **+0.21** | 1.195 | 147.6 | 14.64 GB |

- **A tie** by the 0.4 threshold (+0.21 = 106 images), though everything points the same way:
  holdout loss -0.012, training loss lower at steps 250 (6.421 vs 6.444) and 500 (5.927 vs
  5.951). SSF over plain AdaptFormer: +0.78, versus +0.57 for biases + norms.
- **Cost: 21% lower throughput** (147.6 vs 186.7 img/s) and +6.4 GB. Under a time-capped final
  run that is about 21% fewer steps, which very likely outweighs +0.21. Biases + norms stay in
  the Task 2 recipe; SSF is a report ablation (initialization/design direction) unless a
  time-matched comparison says otherwise.
- Wall time 107.5 min: training ran at a steady ~150 img/s throughout (about 45 min), but the
  final holdout pass ran at 14 img/s (laptop asleep or GPU contention); accuracy unaffected.

## Task 2 SSF: first screen, 1 epoch (2026-10-20)

Is SSF (Lian et al., NeurIPS 2022: a learned per-channel scale and shift after the patch
embedding, every block's norm1/qkv/proj/norm2/fc1/fc2 and the final norm) competitive with
AdaptFormer at a fraction of the parameters? New `mode: ssf`: 205,824 trainable backbone
parameters = **0.24%** (AdaptFormer recipe: 4.27%), 8.40% with the head; mergeable into the
weights after training. Identity init (scale 1, shift 0). Same setup as `af_bn_bias_norm_e1`
(336 px, 40/5 split, prototype cosine head at head lr 4e-3, margin 0.1, smoothing 0.1, batch 256,
bf16, 1 epoch = 1,563 steps); SSF parameters at `lr`, no weight decay. Commit bc26297 (clean),
laptop at 175 W.

| Run | Backbone trainable | Step 0 | Holdout top-1 | vs AF recipe | Holdout loss | Run time / train img/s | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `af_bn_bias_norm_e1` (AF recipe) | 3,672,588 (4.27%) | 42.534 | 72.966 | | 1.207 | 41.9 min / 186.7 | 8.19 GB |
| `af_bn_control_e1` (AF, no bias/norm) | 3,550,476 (4.13%) | 42.534 | 72.396 | -0.57 | 1.235 | 38.1 min / 207.6 | 7.64 GB |
| `ssf_lr3e-3_e1` | 205,824 (0.24%) | 42.534 | **71.902** | **-1.06** | 1.263 | 47.6 min / 159.5 | 13.50 GB |
| `ssf_lr1e-2_e1` | 205,824 (0.24%) | 42.560 | 70.692 | -2.27 | 1.313 | 47.3 min / 159.5 | 13.50 GB |
| `ssf_lr1e-3_e1` | 205,824 (0.24%) | 42.560 | 71.560 | -1.41 | 1.286 | 47.2 min / 159.5 | 13.50 GB |

- With 18x fewer backbone parameters SSF lands 1.06 points (532 images) below the AdaptFormer
  recipe and 0.49 below plain AdaptFormer at 1 epoch. It does not beat it.
- The lr sweep peaks at **3e-3**: 1e-2 is 1.21 worse and 1e-3 (run afterwards, commit c6b8688)
  0.34 worse (a tie by the 0.4 threshold, but the loss agrees: 1.286 vs 1.263). Tuning the rate
  will not close the 1.06 gap to the AdaptFormer recipe.
- **SSF is slower, not faster**: 159.5 vs 186.7 img/s and +5.3 GB. Gradients still flow
  through the whole backbone (SSF follows the patch embedding), and each scale/shift keeps one
  extra activation for its gradient. Under the time-capped final run that means about 15% fewer
  steps than AdaptFormer. Merging removes the cost at inference only.
- Aborted first attempt (`ssf_lr3e-3_e1_aborted_fp32promo`, commit d5ae7d9): fp32 parameters
  times bf16 activations promoted every wrapped output to fp32, 84 img/s and 15.75 GB; stopped
  at step ~300 and fixed in bc26297 (scale/shift in the activation dtype). Its step 0 was 42.56%.

## Prediction-time class-count balancing (2026-10-19)

The holdout, val and test all hold exactly the same number of images per class (5 / 5 / 10),
but argmax predictions pile up (up to 12–13 images in one class, ~15 classes empty). Evaluation
only, CPU, from the saved top-100 softmax stores `<run>/analysis/res_knn/holdout_r384.pt` of
`full_proto_336_len3` and `af_bn_bias_norm_e3` (eval at 384; plain + zoom = summed softmax of
both views). Full holdout, 50,000 images.

- **Sinkhorn (soft):** each class gets a weight `q_c`; alternately row-normalise the reweighted
  softmax and rescale `q_c` so the class's expected count is 5; predict the argmax of `p * q`.
- **Hard:** min-cost matching with exactly 5 slots per class, cost `-log p` over each image's
  top k classes (`scipy.sparse.csgraph.min_weight_full_bipartite_matching`). It has no
  parameters (it does not depend on row/column scaling or temperature).

| Model, views | Argmax | Sinkhorn 1 / 3 / 10 / **50** it | Hard k = 10 / 30 / 100 |
| --- | --- | --- | --- |
| Full FT, plain | 79.026 | 80.200 / 80.676 / 80.890 / **80.924** | 80.092 / 80.212 / 80.242 |
| Full FT, plain + zoom | 79.684 | 80.762 / 81.122 / 81.292 / **81.346** | 80.684 / 80.736 / 80.740 |
| AdaptFormer, plain | 77.542 | 78.176 / 78.630 / 79.074 / **79.246** | 78.220 / 78.312 / 78.370 |
| AdaptFormer, plain + zoom | 78.314 | 79.004 / 79.436 / 79.722 / **79.818** | 78.768 / 78.900 / 78.852 |

- Sinkhorn adds **+1.5 to +1.9** and increases monotonically with iterations (nothing tuned on
  the holdout except picking 50). The hard assignment adds only +0.5 to +1.2: forcing exactly 5
  costs more on confidently right images than it gains.
- Implemented in the group repo as `src/van_balance.py` + `src.predict --balance_per_class N`
  (van-dev bb24049, off by default); it reproduces the table (80.924 / 81.364 for full FT; the
  plain + zoom row differs slightly because only the top 100 of the summed softmax is kept).
- **CodaBench:** the Task 1 final `full_ft_20261017_122237` at crop 0.75 + zoom +
  `--balance_per_class 5`: **0.845** (ID 1002123) vs 0.828 without (ID 995951), **+1.7**,
  matching the holdout. 3,096 of 50,000 val predictions changed; largest class 9, none empty.
- Caveat: this uses the split's known per-class counts; confirm with the Prof before Phase 2.
- Next: Phase 2 test with `--balance_per_class 10`; Task 2 final with balancing.

## Task 2 AdaptFormer: prediction views re-tuned at 448 (2026-10-19)

Flip and zoom added less on CodaBench at 448 (over plain: flip +0.2, zoom +0.5; at 384 zoom
was +1.3), so the view settings tuned at 336 were re-checked. Evaluation only,
`scripts/eval_zoom.py` on `af_res_screen_448/last.pt` (the only 448 AdaptFormer with a holdout;
10 images/class, 62.72% at crop 0.9), full holdout (50,000), batch 32, 4 workers, laptop at
175 W, about 35 min per run. Maps `cam`, 2x copy; masses 0.5 / 0.7 / 0.9 in each run; settings
chosen on the even half, reported on the odd half. Outputs:
`af_res_screen_448/analysis/tta_sweep_448/`. The plain views match `eval_tta.py` (62.902 vs
62.906 at 0.75, 62.716 vs 62.724 at 0.9; bf16).

| Zoom crop / min box | View (passes) | Mass 0.5 even / odd | **Mass 0.7** even / odd | Mass 0.9 even / odd |
| --- | --- | --- | --- | --- |
| 0.75 / 8 (current) | plain + zoom (2) | 63.580 / 62.796 | 63.552 / **63.244** | 63.684 / 63.228 |
| 0.75 / 11 | plain + zoom (2) | 63.632 / 62.840 | 63.556 / 63.244 | 63.684 / 63.228 |
| **0.9 / 8** | plain + zoom (2) | 63.984 / 63.288 | **64.116 / 63.520** | 63.868 / 63.172 |
| 1.0 / 8 | plain + zoom (2) | 63.952 / 63.408 | 63.996 / 63.460 | 63.648 / 62.916 |
| 0.9 / 8 | + flips (4) | 64.072 / 63.408 | 64.224 / 63.568 | 64.024 / 63.176 |

Without zoom (even / odd): crop 0.75 plain 63.192 / 62.612, + flip 63.316 / 62.780; crop 0.9
plain 63.048 / 62.384, + flip 63.280 / 62.492; crop 1.0 plain 62.512 / 62.104.

- **Chosen on the even half: crop 0.9, mass 0.7** (plain + zoom). On the odd half +0.28 over
  the current crop 0.75 (paired z 1.7; all 50,000: +0.42, z 3.6, but that includes the half it
  was chosen on). The tighter 0.75 crop helps the plain view (+0.2) but hurts the zoom: at 448
  the zoom already enlarges the subject, and cropping first removes context the box needs. This
  matches the full-FT screen, where the crop gain disappeared at 448.
- Mass 0.7 stays (best on even at crops 0.9 and 1.0). The minimum box (8 vs 11 patches)
  changes almost nothing: the 0.7-mass box is rarely below 11 patches.
- 4 passes add only +0.05 on the odd half over the 2-pass view: not worth double the cost.
- Even on this weaker screen model, zoom still adds about +1.1 over plain at crop 0.9; on
  CodaBench the final gained +0.5. Expect a CodaBench gain of +0.1 to +0.3 from the crop change.
- **CodaBench (2026-10-19): 0.814** with crop 0.9 + zoom (ID 1002322) vs **0.809** with crop 0.75
  + zoom (ID 1001880): **+0.5**, more than the holdout's +0.28 on the odd half (the all-holdout
  +0.42 was closer). Same `last.pt` of `van_adaptformer_20261018_073912`, team `src.predict` at
  HEAD d6be092 (`--crop_pct 0.9 --zoom`, mass 0.7), predicted on the laptop (batch 32, 4 workers);
  45,234 of 50,000 predictions agree with the crop 0.75 file. Task 2 best (without class
  balancing).
- **Full FT at 448 (2026-10-20): 0.832** with crop 0.9 + zoom (ID 1004167; `full_ft_20261017_122237`,
  same team `src.predict`, laptop, no balancing) vs **0.828** with crop 0.75 + zoom (ID 995951):
  **+0.4**. The crop change chosen on the AdaptFormer screen carries over to full FT.

## Task 2 AdaptFormer: depth-graded adapter widths, 1 epoch (2026-10-18)

Does the same budget work better spread by depth? Every run's learned branch scales grow with
depth, which suggested deep blocks want more adapter capacity. New option: `adapter_dim` may be a
list of 12 widths (one per block; an int behaves as before; PCA init uses each adapter's own
width). `af_width_graded_e1` = `af_bn_bias_norm_e1` with widths 96 / 192 / 288 in blocks 1-4 /
5-8 / 9-12: same mean width 192, so the same 3,672,588 trainable backbone parameters (4.27%).
Commit f38a5bc (clean), 336 px, 40/class, 1 epoch (1,563 steps), laptop at 163-175 W, keep-awake.

| Run | Widths | Step 0 | Holdout top-1 | Diff | Holdout loss | Run time / mean img/s |
| --- | --- | --- | --- | --- | --- | --- |
| `af_bn_bias_norm_e1` (control) | 192 x 12 | 42.534 | 72.966 | | 1.207 | 41.9 min / 186.7 |
| `af_width_graded_e1` | 96 x 4, 192 x 4, 288 x 4 | 42.534 | 72.894 | **-0.07** | 1.202 | 42.9 min / 183.3 |

Learned branch scales after 1 epoch (blocks 1-12):

- Control: 0.017, 0.031, 0.034, 0.058, 0.060, 0.082, 0.092, 0.112, 0.098, 0.080, 0.071, 0.147.
- Graded: 0.016, 0.033, 0.040, 0.062, 0.063, 0.083, 0.092, 0.106, 0.090, 0.069, 0.065, 0.140.

- **A tie** (-0.07, 36 images; loss -0.004; training loss identical at steps 250 and 500).
  Where the adapter capacity sits does not matter at this budget.
- The scales barely move when the widths change (within 0.011 in every block). They measure how
  much each block's output should shift, not a shortage of width there, so they were the wrong
  signal for reallocating capacity. This agrees with the fit check: AdaptFormer is not
  capacity-limited.
- Uniform width 192 stays. Not run: "adapters only in the last N blocks" (cheaper steps, more
  epochs per budget); after this tie its case rests only on speed.

## Task 2 AdaptFormer: regularization screen, 3 epochs (2026-10-17/18)

Follow-up to the fit check below (17.5-point train-vs-holdout gap, generalization rather than
capacity). Two arms against `af_bn_bias_norm_e3` (76.756%), same recipe, split, seed, 336 px,
40/class, 3 epochs (4,689 steps), laptop at 175 W with a keep-awake request. Commit 05ef616
(clean), configs `af_aug_ra_re_e3.yaml` and `af_adrop01_e3.yaml`. Holdout top-1 at crop 0.9.

| Epoch | Control `af_bn_bias_norm_e3` | `af_aug_ra_re_e3` (RandAugment m9 + erasing 0.25) | `af_adrop01_e3` (adapter dropout 0.1) |
| --- | --- | --- | --- |
| 0 (gate) | 42.534 | 42.534 | 42.534 |
| 1 | 69.616 | 68.096 (-1.52) | 69.688 (+0.07) |
| 2 | 75.430 | 74.952 (-0.48) | 75.482 (+0.05) |
| 3 | **76.756** | **76.436 (-0.32)** | **76.722 (-0.03)** |
| Holdout loss, epoch 3 | 1.195 | 1.176 | 1.192 |
| Run time / mean img/s / 48 GB-h | 118.4 min / 186.6 / 0.655 | 119.9 min / 184.4 / 0.663 | 117.9 min / 187.3 / 0.652 |

- Both arms tie with the control (inside the 0.4-point threshold); neither replaces the recipe.
- In timm, `auto_augment` replaces the default colour jitter, so arm 1 also drops jitter.
  It starts 1.5 points behind and closes the gap every epoch (-1.5, -0.5, -0.3) with the
  lowest holdout loss: the usual slow-start pattern, so it may pay off over the finals' ~9
  epochs, which this screen cannot show. Training loss is higher throughout (6.46 vs 6.03 at
  step 500).
- Adapter dropout 0.1 changes nothing measurable at any epoch: the adapters are not where the
  overfitting is controlled.
- The first launch of arm 1 crashed in the step-zero evaluation (cuBLAS
  `CUBLAS_STATUS_INTERNAL_ERROR`, then an abort in the pinned allocator) and left 12 orphaned
  loader workers (~11 GB). The identical relaunch ran cleanly; the partial output is kept as
  `af_aug_ra_re_e3_CRASH1`.

## Task 2 vs Task 1: train-vs-holdout fit check (2026-10-17)

Is AdaptFormer's gap to full FT due to too little capacity (it cannot fit the training set)
or to generalization? Evaluation only, `scripts/eval_fit.py`, `best.pt` (step 4,689) of
`af_bn_bias_norm_e3` and `full_proto_336_len3` (both 336, 3 epochs, 40/class, same split),
plain view at crop 0.75, bf16, laptop RTX 5080 at 175 W, about 2 min per 50,000 images. Train
sample = the first 5 images per class of `internal_train.txt` (50,000, all seen in training);
CE is plain cross-entropy on the eval logits (no margin). Gate: the holdout reproduces
`eval_zoom.json` crop 0.75 plain (77.448 vs 77.436; 78.814 vs 78.798, bf16 noise).

| Run | Train5 top-1 / top-5 / CE | Holdout top-1 / top-5 / CE | Gap (top-1) |
| --- | --- | --- | --- |
| AdaptFormer `af_bn_bias_norm_e3` (4.27% backbone) | 94.97 / 98.63 / 0.212 | 77.45 / 91.01 / 1.161 | 17.5 |
| Full FT `full_proto_336_len3` (100%) | 95.46 / 99.19 / 0.314 | 78.81 / 93.25 / 1.021 | 16.7 |

- With 4.27% of the backbone trainable, AdaptFormer fits the training images almost as well as
  full FT (-0.5 top-1, and a lower train CE because of the CosFace margin's sharper logits). The
  holdout gap (-1.37) is about 3x the training gap, so most of it is generalization, not capacity.
- Both runs overfit strongly after only 3 epochs (a 17-point gap); the finals train about 9
  epochs on 45/class, so their training fit is probably higher still.
- Consequence: adding capacity by combining PEFT methods (e.g. + LoRA on attention) is unlikely
  to close much of the gap. The better levers are regularization or features that transfer better
  (resolution, augmentation, adapter dropout, the AdaptFormer recipe's loss/init).

## Task 2 AdaptFormer: training resolution at equal compute (2026-10-16)

Does the 384 gain of full FT carry over to a mostly frozen backbone (frozen 384 features are
weaker: prototypes 40.47% vs 42.53% at 336)? Recipe = `af_bn_bias_norm_e3` (the Task 2 final:
width 192, norm input, PCA init, learned scale, CosFace 0.1, trainable biases + LayerNorms;
4.27% of the backbone at 336, 4.266% at 384). Setup = the full-FT resolution screen: 10 images/class,
0.48 x 48 GB-hours (1.45 h), stopped by the clock, no validation during training, `last.pt` scored
with `eval_tta.py`. Configs `af_res_screen_336/384.yaml`, commit fb9e693, both arms clean.

| Arm | Epochs (steps) | Train img/s | Peak GB | Crop 0.75 | **Crop 0.9** | Crop 1.0 | All crops + flip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `af_res_screen_336` | 9.4 (3,665) | 186 | 8.20 | 60.63 | **59.63** | 58.73 | 61.27 |
| `af_res_screen_384` | 6.7 (2,627) | 127 | 10.46 | 61.72 | 61.18 | 60.42 | 62.61 |
| `af_res_screen_448` (2026-10-18) | 4.7 (1,850) | 96 | 7.37* | **62.91** | **62.72** | **62.31** | **63.98** |

\* Batch 32 x 8 instead of 64 x 4 (same effective 256), so the peak is not comparable.

- **384 wins by +1.55 at crop 0.9 and +1.09 at crop 0.75**, more than for full FT (+0.86 / +0.42),
  despite 29% fewer epochs and weaker starting features. The adapters, biases and LayerNorms adapt
  the frozen backbone to the higher resolution.
- **448 (2026-10-18, commit 29dd19a, clean)**: **+1.54 over 384 at crop 0.9**, +1.19 at crop
  0.75, +1.37 for all crops + flip, with 30% fewer epochs (4.7 vs 6.7). Curve at crop 0.9:
  59.63 / 61.18 / 62.72 for 336 / 384 / 448, i.e. +1.55 then +1.54. Unlike full FT (+0.86 then
  +0.73, flat by 512), AdaptFormer's returns have not shrunk yet. Clean run: 175 W, keep-awake
  request, largest gap between log rows 177 s, 0.480 normalized 48 GB-hours, stopped by the
  clock with the lr at zero. PCA init used 16 batches of 32 (half the images of the 384 arm).
  Scored with `eval_tta.py --crop-pct 0.75 0.9 1.0` (`analysis/eval_tta.json`).
- **CodaBench at 448 (2026-10-18)**: team trainer, `configs/adaptformer_w192_448px_bias-and-layernorm_fused-adamw_compile_2.4h.yaml`
  (cs604-main van-dev d65cbc3), run `CS604_RUNS/van_adaptformer_20261018_073912`, A100 40 GB,
  144 min, 6.1 epochs, ~317 img/s, batch 64 x 4, 13.76 GB peak, Qian's `prototypes_448.pt`.
  **0.804 plain** (ID 1000827; 384: 0.794, +1.0), **0.806 crop 0.75 + flip** (ID 1000837),
  **0.809 crop 0.75 + zoom** (ID 1001880; 384: 0.807, +0.2). Task 2 final candidate. As for full
  FT, prediction-time views add less at higher resolution (zoom over plain: +1.3 at 384, +0.5 at
  448). The Colab zoom prediction stopped at 31,062 of 50,000 rows; the submitted file was
  predicted on the laptop (team `src.predict`, batch 32, 4 workers; batch 64 x 12 workers crashed
  at 8,064 rows) and agrees with the partial Colab rows on 99.5% of images.
- The first 384 run is invalid and kept as `af_res_screen_384_INVALID_sleep`: the laptop slept
  18:53-19:44 during it (Windows Kernel-Power 42/107), the clock-fitted budget kept counting, so
  it trained 2.7 epochs (58.25% at crop 0.9). The rerun held a keep-awake request
  (SetThreadExecutionState) from 5 minutes in; its largest gap between log rows was 133 s.
- Next: Colab run of `configs/adaptformer_w192_384px_bias-and-layernorm_fused-adamw_compile_2.4h.yaml` (cs604-main van-dev 2bd8e78),
  against 0.802 (336, crop 0.75 + zoom).
- **CodaBench (2026-10-17)**: run `CS604_RUNS/van_adaptformer_20261016_142837`, A100 40 GB, 144 min,
  8.6 epochs, 19.87 GB peak, all 45/class. **0.794 plain** (ID 994764), **0.807 crop 0.75 + zoom**
  (ID 994759; 336: 0.802, +0.5). Crop 0.75 + flip not submitted (336: 0.794). The screen's +1.55
  shrinks to +0.5 with zoom on CodaBench, as for full FT (+0.5 zoom, +1.0 plain). Task 2 final
  candidate. Latency 9.86 ms at batch 1 on the A100.

## Task 1 full fine-tune: neighbour label smoothing (2026-10-16)

Motivation (`scripts/analyze_errors.py` on `full_proto_336_len3/best.pt`, crop 0.75, 78.81%
top-1, 93.25% top-5): 45.8% of holdout errors predict a class within +-10 ids of the truth
(16.9% within +-1; 0.14% for a random class), and the most-confused pairs are adjacent ids, so
class ids appear to follow the iNat taxonomy. When wrong, the true class is ranked 2nd in 40% of
cases. Accuracy depends little on aspect ratio or image size; confidence is well calibrated.

`nbr_smooth_384_w5` (commit cbd421b) = `res_screen_384` with the same 0.1 label-smoothing mass
moved from all classes onto the 10 ids within +-5 of the target (`neighbour_smoothing_window: 5`,
`neighbour_smoothing_share: 1.0`). Same data, budget (1.45 h, stopped by it), seed and scoring.

| Arm | Steps | Crop 0.75 | **Crop 0.9** | Crop 1.0 | All crops + flip |
| --- | --- | --- | --- | --- | --- |
| `res_screen_384` (uniform smoothing) | 2,146 | 64.28 | **64.06** | 63.21 | 65.05 |
| `nbr_smooth_384_w5` | 2,227 | 64.04 | **63.92** | 63.32 | 65.12 |

- **No effect**: -0.14 at crop 0.9, -0.24 at 0.75, +0.11 at 1.0, all within seed noise (~0.3).
  Peak memory equal (7.74 GB); 4% more steps from throughput variation.
- Reading: the model already ranks taxonomic neighbours high (that is why its errors cluster);
  pushing probability onto them does not help separate them, which is what top-1 needs.
  Taxonomy-aware losses tend to reduce mistake severity rather than raise top-1.
- Next: drop it. No share-0.5 follow-up.

## Task 1 full fine-tune: training resolution at equal compute (2026-10-15/16)

Question: is 336 the best training resolution once the epoch count it costs is paid for?
Three arms, same recipe as the final full FT (prototype cosine head, lr 4e-4, layer decay 0.75,
head lr 4e-3, wd 0.05, label smoothing 0.1, drop_path 0.1, clip 1.0, effective batch 32 x 8 =
256, bf16), each given the same budget: 0.48 x 48 GB-hours = 1.45 h on the laptop RTX 5080
(175 W), schedule fitted to the clock and stopped by it. A scale-down of the A100 final
(~8.5 epochs at 336): 10 training images/class (100,000), fixed 5/class holdout (50,000,
split seed 604). No validation during training (it would cost the arms unequal time);
`last.pt` scored afterwards with `scripts/eval_tta.py`. Configs `res_screen_*.yaml`, prototypes
from 40/class (`prototype_init_384.yaml`: 40.47% at 384; 42.53% at 336). Commit 156f02a
(336 arm: b3fb2be + the same uncommitted change, committed as 156f02a before the other arms).
Progressive arm: `train_low_image_size: 224`, `train_low_fraction: 0.75` (new key, 156f02a).

| Arm | Epochs (steps) | Train img/s | Peak GB | Final train loss | Crop 0.75 | **Crop 0.9** | Crop 1.0 | All crops + flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `res_screen_336` | 7.7 (3,010) | 152 | 6.31 | 1.744 | 63.86 | **63.20** | 62.51 | 64.55 |
| `res_screen_384` | 5.5 (2,146) | 108 | 7.74 | 1.839 | **64.28** | **64.06** | 63.21 | **65.05** |
| `res_screen_448` (2026-10-16/17) | 4.1 (1,604) | 80 | 9.85 | 1.930 | **64.83** | **64.79** | **64.37** | **65.77** |
| `res_screen_512` (2026-10-17) | 2.9 (1,141) | 56 | 12.32 | 2.156 | 64.78 | **64.98** | **64.63** | **65.86** |
| `res_screen_224to384` | 13 at 224 + 1.3 at 384 (5,584) | 355 / 112 | 7.73 | 1.639 | 59.85 | **59.74** | 59.05 | 60.58 |

Plain single-view accuracy unless noted. Trainable: 93.67M at 336, 93.77M at 384 (100%; the
384 position embedding is larger), of which the head is 7.68M. All three stopped by the budget
(0.482 normalized 48 GB-hours each).

- **384 beats 336 by +0.86 at crop 0.9 and +0.42 at crop 0.75** with 30% fewer epochs and
  weaker starting prototypes. Single seed; seed noise was about 0.3, so this is a small gain,
  far below 224 -> 336 (+5). Above 336 the returns are diminishing.
- At 384 the tighter crop matters less (+0.2 vs +0.7 at 336): the extra resolution already
  supplies the detail crop 0.75 was buying.
- 384 costs 1.4x per training image (108 vs 152 img/s) and about the same at inference.
- **Progressive 224 -> 384 loses by 4.3 points** despite the most epochs and the lowest train
  loss. Two causes this screen cannot separate: (1) the 384 phase falls in the last 23% of
  the cosine schedule, where lr is at most ~14% of peak, so 1.3 epochs barely adapt the model
  to 384; (2) ~14 epochs on 10 images/class overfit (lowest train loss, worst holdout). Cause
  (2) is a screen artefact (the final has 45/class) but cause (1) would carry over. Not
  scored at 224, so how much of the gap is the resolution switch itself is unknown.
- Caveat for all arms: 10 images/class with a full-length schedule may favour fewer epochs, which
  could flatter 384 over 336 somewhat; the final has 4.5x the data per epoch.
- Next: 384 is the candidate final resolution (expected about +0.5 to +1 on CodaBench over
  0.807, unconfirmed). Drop progressive resizing in this form. The planned hyperparameter
  round (lighter drop_path / label smoothing, peak lr) should run at 384.
- **448 (2026-10-16/17, commit a451fac)**: still ahead of 384 by **+0.73 at crop 0.9**, +0.55 at
  crop 0.75 and +1.16 at crop 1.0, with 4.1 epochs (vs 5.5) and frozen prototypes at only 37.20%
  (`prototype_init_448.yaml`; 40.47% at 384). Curve at crop 0.9: 63.20 -> 64.06 -> 64.79, i.e.
  +0.86 then +0.73: returns shrink only slightly, despite 79% of holdout images having a short side
  of 300-400 px (448 mostly upsamples, giving the 16 px patches finer detail per object). The
  crop gain disappears at 448 (0.75 vs 0.9: +0.04, vs +0.66 at 336). Largest log gaps 217 s, all
  at epoch boundaries. The first prototype build failed on a one-off read error (OSError 22) in a
  loader worker during holdout scoring; all 50,000 holdout images open fine; rebuilt with 8 workers.
- **512 (2026-10-17, commit 80914f6)**: **flat**. +0.19 over 448 at crop 0.9 (64.98 vs 64.79),
  -0.05 at crop 0.75, +0.26 at crop 1.0, +0.09 all crops + flip: all within seed noise (~0.3),
  with 2.9 epochs (vs 4.1) and frozen prototypes at 33.85% (`prototype_init_512.yaml`; 37.20% at
  448). Curve at crop 0.9: 63.20 / 64.06 / 64.79 / 64.98 for 336 / 384 / 448 / 512, gains +0.86,
  +0.73, +0.19. No log gaps over 300 s. Returns flatten between 448 and 512 on this screen. The
  screen (10 images/class) tolerates few epochs better than a 45/class final, which argues for 448
  over 512 on Colab.
- **CodaBench at 448 (2026-10-17)**: team trainer, `configs/full_ft_448px_proto-cosine_llrd0.75_fused-adamw_compile_2.4h.yaml` (cs604-main
  `van-dev` 2013466), run `CS604_RUNS/full_ft_20261017_122237`, A100 40 GB, 144 min, 5.5 epochs,
  batch 64 x 4, 17.80 GB peak, Qian's `prototypes_448.pt`. **0.825 plain** (ID 995914; 384: 0.817,
  +0.8, the screen predicted +0.73), 0.827 crop 0.75 + flip (ID 995918, uploaded under a 384 file
  name but it is this run's prediction), **0.828 crop 0.75 + zoom** (ID 995951; 384: 0.824).
  Latency 7.16 ms at batch 1. Task 1 final. The screen tracked CodaBench plain at both steps.
- **CodaBench confirmation (2026-10-16)**: team trainer, `configs/full_ft_384px_proto-cosine_llrd0.75_fused-adamw_compile_2.4h.yaml`
  (cs604-main `van-dev` 7e05162), run `CS604_RUNS/full_ft_20261016_052544`, A100 40 GB, 144 min,
  7.9 epochs, 25.34 GB peak, all 45/class, prototypes from all 45/class (team script). **0.817
  plain** (ID 992465) vs 0.807 at 336 (+1.0, the screen predicted +0.86); 0.821 crop 0.75 + flip
  (ID 992482; 336: 0.813 at crop 0.7); **0.824 crop 0.75 + zoom** (ID 992449; 336: 0.819).
  Latency 7.02 ms at batch 1 on the A100. Task 1 final candidate.

Evaluation only, `scripts/eval_res_knn.py` (commits d9e6f2b, c437538), full internal holdout
(50,000), laptop RTX 5080 at 175 W, `best.pt` of AdaptFormer `af_bn_bias_norm_e3` and full FT
`full_proto_336_len3` (both trained at 336 for 3 epochs). Eval crop 0.75, zoom `cam` m0.7
throughout. Reference = the current submission view, 336 plain + zoom (2 passes). Gate: the
336 plain and plain + zoom views reproduce `eval_zoom.json` exactly (77.436 / 78.476 and
78.798 / 79.588). Settings chosen on the even half, reported on the odd half; z is paired
against the reference over all 50,000 images. Outputs: `<run>/analysis/eval_res_knn.json`,
stores in `analysis/res_knn/`.

**Higher resolution** (position embeddings resampled bicubically, same field of view):

| View | Passes | AdaptFormer all / odd | Full FT all / odd |
| --- | --- | --- | --- |
| 336 plain (trained size) | 1 | 77.436 / 77.216 | 78.798 / 78.544 |
| 384 plain | 1 | 77.542 / 77.380 | 79.026 / 78.904 |
| 448 plain | 1 | 76.672 / 76.384 | 78.676 / 78.568 |
| **336 plain + zoom (reference)** | 2 | **78.476 / 78.412** | **79.588 / 79.404** |
| 384 plain + zoom | 2 | 78.314 / 78.228 | 79.684 / 79.664 |
| 384 plain + 336 zoom | 2 | 78.434 / 78.380 | 79.700 / 79.616 (z 1.5) |
| 448 plain + 336 zoom | 2 | 78.310 / 78.228 | 79.594 / 79.492 |
| 336 plain + zoom + 384 plain | 3 | 78.732 / 78.544 | 79.990 / 79.876 |
| 336 plain + zoom + 384 plain + zoom | 4 | 78.850 / 78.700 | 80.056 / 79.916 |

- 384 alone is +0.1 (AdaptFormer) and +0.2 (full FT) over 336 plain; 448 is worse for both,
  and zoom views get worse with size (the zoom already enlarges the subject).
- At 2 passes resolution adds nothing to the reference (AdaptFormer -0.04, full FT +0.11,
  z 1.5, not significant). Adding 384 views as extra passes gives +0.3 to +0.5, i.e. a
  multi-scale ensemble; 4 passes of flips/zoom earlier gave +0.04 to +0.13.

**kNN over the run's own 400,000 training images** (one forward pass, eval view at 336;
pooled normalized features; neighbour votes `exp(cos / T)`; blend
`(1 - lam) p_classifier + lam p_knn`; query `plainzoom` = mean of the plain and zoom
features, so the holdout needs no extra pass):

| | AdaptFormer | Full FT |
| --- | --- | --- |
| kNN alone, best (`plainzoom`) | 73.60 (k100, T 0.05) | 76.66 (k20, T 0.05) |
| Chosen on even | `plainzoom` k10 T0.1 lam0.5 | `plainzoom` k10 T0.05 lam0.4 |
| Reference + kNN, all / odd | **79.510 / 79.360** | **80.642 / 80.436** |
| Gain on the odd half | **+0.95** | **+1.03** |
| Paired z (all) | 13.2 | 10.4 |
| Full FT: best 2-pass res view (384 plain + 336 zoom) + kNN (`plain` k10 T0.05 lam0.3) | | 80.862 / 80.676 (+1.27, z 13.3) |

- Odd half, chosen k and T, by lam: AdaptFormer 0.1-0.6: 78.58, 78.72, 78.99, 79.24, 79.36,
  79.23; full FT: 80.06, 80.40, 80.54, 80.44, 80.12, 79.57. Broad optimum at lam 0.3-0.5;
  every lam in 0.1-0.5 beats the reference.
- The kNN alone is 5 (AdaptFormer) / 3 (full FT) points below the classifier but errs on
  different images, so the blend gains about +1 point for one extra inference pass over the
  training set and no extra holdout passes.
- Cost on the laptop: training-image bank pass 13.5 min (AdaptFormer) / 12.5 min (full FT)
  for 400,000 images (615 MB fp16); retrieval is negligible. 448 + 2x zoom needed batch 32 and
  4 workers (batch 64 x 8 workers ran out of commit/GPU memory at 896 px).
- Caveats for the final models: they train on all 45/class for longer (about 16 epochs full FT
  on the A100, vs 3 here), so their training features may be tighter and the best lam and T
  can shift; the holdout setup matches theirs only in that the bank = the model's own training
  images. Whether the bank pass counts toward the training budget, and whether it is allowed,
  is a question for the Prof before any CodaBench use.

## Task 2 AdaptFormer + trainable biases and LayerNorms: learning-rate screen (2026-10-11)

1 epoch against `af_bn_bias_norm_e1` (72.966%, adapters and biases/norms both at lr 1e-3),
same setup. Commit 57356a7 (clean; new opt-in key `bias_lr`); arm (b) ran the next morning
at commit 14526aa (records only).

| Run | Change | Step 0 | Holdout top-1 | Diff | Holdout loss | Run time |
| --- | --- | --- | --- | --- | --- | --- |
| `af_bn_biaslr3e-3_e1` | `bias_lr: 3e-3` (adapters stay 1e-3) | 42.53% | 72.660% | -0.31 | 1.210 | 47.8 min* |
| `af_bn_lr3e-3_e1` | `lr: 3e-3` (adapters and biases/norms) | 42.53% | 72.860% | -0.11 | 1.209 | 41.0 min |

\* The charger was unplugged at about step 1,350 (GPU capped at 60 W, 193 -> 96 img/s);
accuracy is unaffected.

- A higher rate for the biases/norms is a tie that leans worse (-0.31, loss equal): they keep
  the adapter rate. The team trainer needs no `bias_lr`.
- One rate of 3e-3 for everything also ties (-0.11, loss equal). Both arms tie, so the
  recipe keeps lr 1e-3 (tie rule: the lower rate). The learning rate is settled for this
  recipe; `configs/adaptformer_w192_336px_bias-and-layernorm_cosface0.1_2.4h.yaml` (group repo) stays at 1e-3.

## Task 2 AdaptFormer + trainable biases and LayerNorms: phase 2, 3 epochs (2026-10-11)

`af_bn_bias_norm_e3` (commit e0b1d54, clean): the `af_e_combined_e3` recipe for 3 epochs
(4,689 steps) plus `adapter_train_bias` and `adapter_train_norm`; backbone 3,672,588 = 4.27%
(11.68% with the head). Same split, seed, laptop at 175 W. The control `af_e_combined_e3` ran on
2026-10-08 at commit bd1bb63.

| Epoch | `af_e_combined_e3` | `af_bn_bias_norm_e3` | Diff | Holdout loss (control / new) |
| --- | --- | --- | --- | --- |
| 0 (gate) | 42.534% | 42.534% | 0 | 3.696 / 3.696 |
| 1 | 69.168% | 69.616% | +0.45 | 1.411 / 1.399 |
| 2 | 74.848% | 75.430% | +0.58 | 1.208 / 1.177 |
| 3 | 76.258% | **76.756%** | **+0.50** | 1.226 / 1.195 |

Run time 118.4 min vs 113.8 (mean 186.6 vs 196.0 img/s, on different days), peak 8.19 vs
7.64 GB, 0.655 normalized 48 GB-hours.

Prediction views (`scripts/eval_zoom.py --maps cam --mass 0.7`, `best.pt` = step 4,689, full
holdout; the crop 0.9 plain gate reproduced 76.762 vs 76.756). Paired over the same 50,000
images against the control's `eval_zoom_images.pt`:

| View | Passes | `af_e_combined_e3` | `af_bn_bias_norm_e3` | Diff | Paired z |
| --- | --- | --- | --- | --- | --- |
| crop 0.9 plain | 1 | 76.280 | 76.762 | +0.48 | 4.6 |
| crop 0.75 plain | 1 | 76.938 | 77.436 | +0.50 | 4.9 |
| crop 0.75 + flip | 2 | 77.100 | 77.690 | +0.59 | 6.0 |
| **crop 0.75 + zoom, `cam` m0.7** | 2 | 77.946 | **78.476** | **+0.53** | 5.2 |
| crop 0.75 + flip + zoom + zoom-flip | 4 | 78.076 | 78.572 | +0.50 | 4.9 |

- The 1-epoch gain (+0.57) holds at 3 epochs (+0.50) and adds on top of the zoom (+0.53).
  The paired z measures image-level noise only; one seed, so seed spread (~0.33) still applies,
  but the gain is the same sign and size at every epoch and in every view.
- **CodaBench (2026-10-12): 0.802** with crop 0.75 + zoom (ID 980309), +0.5 over the final
  without biases/norms (0.797, ID 972126), as the holdout predicted (+0.53); **0.794** with crop 0.75 + flip (ID 980317; 0.790
  without), so zoom beats flip by +0.8 (holdout +0.79). Final run:
  `CS604_RUNS/van_adaptformer_20261012_060248`, group repo `configs/adaptformer_w192_336px_bias-and-layernorm_cosface0.1_2.4h.yaml`
  (van-dev), A100 40 GB, 144 training minutes, batch 128 x 2, ~462 img/s, 8.9 epochs (the
  personal-pipeline final `af_final_45` did ~515 img/s, ~10 epochs, at batch 256 x 1); peak
  15.61 GB; A100 batch-1 latency 10.28 ms. Training loss rose over epochs 2-6, then fell to 4.57
  (margin x learned scale).
- Not tested: equal wall-clock time. The bias/norm gradients cost about 10% throughput on the
  laptop (same-session 1-epoch arms: 186.7 vs 207.6 img/s), so a time-capped final run gets
  about 10% fewer steps. A `max_steps: 4220` run (90% of 4,689, cosine fitted to it) would
  measure that directly.

## Task 2 AdaptFormer + trainable biases and LayerNorms: phase 1 (2026-10-11)

Question: does training the pretrained backbone's biases (BitFit) and LayerNorm weights beside
the adapters help? New opt-in keys `adapter_train_bias` (all Linear, patch-embedding and
LayerNorm biases: 83,712 + 19,200) and `adapter_train_norm` (LayerNorm weights: 19,200); they
train at the adapter lr 1e-3 with no weight decay. Common setup: the `af_e_combined_e3` recipe
(336 px, width 192, norm2 input, PCA init, learned scale, prototype cosine head at head lr 4e-3,
margin 0.1, smoothing 0.1, wd 0.05, batch 256, bf16) for **1 epoch** (1,563 steps), fixed 40/5
split (seed 604), seed 0. Commit 67d59e3 (clean). RTX 5080 laptop at 175 W, run one after
another in the same session.

| Run | Trains besides adapters + head | Backbone trainable | Step 0 | Holdout top-1 | Holdout loss | Train img/s | Run time | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `af_bn_control_e1` | nothing | 3,550,476 (4.13%) | 42.53% | 72.396% | 1.235 | 207.6 | 38.1 min | 7.64 GB |
| `af_bn_bias_e1` | biases | 3,653,388 (4.25%) | 42.53% | 72.836% (+0.44) | 1.211 | 191.9 | 40.7 min | 8.19 GB |
| `af_bn_bias_norm_e1` | biases + LN weights | 3,672,588 (4.27%) | 42.53% | **72.966% (+0.57)** | 1.207 | 186.7 | 41.9 min | 8.19 GB |

- The step-zero gate passed on every arm (42.534%): trainable biases/norms start at their
  pretrained values. Total trainable with the head: 11.55% / 11.66% / 11.68%.
- Both arms beat the control by more than the 0.4-point tie threshold; the holdout loss agrees.
  LayerNorm weights add +0.13 over biases alone (a tie). Phase-2 candidate:
  `af_bn_bias_norm_e1`.
- **Cost: 10% lower throughput** (186.7 vs 207.6 img/s) and +0.55 GB. At equal epochs the
  gain is +0.57; under the time-capped final run it buys about 10% fewer steps, so phase 2
  should be judged against that (3 epochs vs `af_e_combined_e3`, 76.258%).
- PCA init calibrates `down` on norm2(x) before training; in arm (c) norm2's weight trains, so
  that input drifts. It did not hurt at 1 epoch.
- Infrastructure: the first attempts (commit e7e2eb4, folders `*_crashed_commitlimit`)
  crashed: the control at step ~500 (DataLoader "Couldn't open shared file mapping",
  error 1455) and arm (b) right after its gate (0xC0000409). Windows logged low virtual
  memory, with the trainer at 26 GB of commit. The PCA calibration loader's 12 persistent
  workers stayed alive beside the training and eval workers; commit 67d59e3 shuts them down
  after the init (results unaffected), after which all three arms completed at about 101 GB
  of commit.

## Prediction-time "look, then zoom in" (2026-10-09)

Evaluation only, `scripts/eval_zoom.py` (uncommitted at the time of the runs), laptop RTX 5080
at 175 W, full internal holdout (50,000 images). The plain view is classified; a saliency map on
the 21 x 21 patch grid picks the smallest square window (at least 8 patches) holding `mass` of
the map; that window is cut from a 2x-resolution copy of the same view, resized to 336 and
classified; softmax averaged with the plain view (2 passes). Maps: `attn` = last block's CLS
attention (mean over heads); `cam` = cosine of each final patch token with the predicted class's
head weight, minus its mean, clamped at 0; `center` = control, the `cam` box's size moved to the
centre (zoom without location). Eval crop 0.75 throughout. Gates: crop 0.9 plain reproduces
76.280 (run 76.258) and 78.388 (run 78.388). Outputs: `<run>/analysis/eval_zoom*.json`.

| View (crop 0.75) | Passes | AdaptFormer `af_e_combined_e3` | Full FT `full_proto_336_len3` |
| --- | --- | --- | --- |
| plain | 1 | 76.938 | 78.798 |
| plain + flip (current submission) | 2 | 77.100 | 78.970 |
| plain + zoom, `center` m0.7 (control) | 2 | 77.666 | 79.434 |
| plain + zoom, `attn` m0.85 | 2 | 77.822 | 79.492 |
| **plain + zoom, `cam` m0.7** | 2 | **77.946** | **79.588** |
| crop 0.9 + crop 0.75, each + flip | 4 | 77.898 | 79.632 |
| plain + flip + zoom + zoom-flip, `cam` m0.7 | 4 | 78.076 | 79.630 |

- **`cam` m0.7 zoom beats plain + flip at the same 2 passes**: +0.85 (AdaptFormer, paired
  z = 9.2) and +0.62 (full FT, z = 6.9). Chosen on the even half, it holds on the odd half
  (+0.92 / +0.65). It matches the 4-pass multi-crop at half the passes.
- **Location matters, but it is the smaller part.** `cam` vs the centred control:
  +0.28 (AdaptFormer, z = 3.2), +0.15 (full FT, z = 1.8, not significant). Zoom alone
  is far better from the `cam` box (76.06 vs 74.49 AdaptFormer; 77.82 vs 76.58 full FT):
  the saliency box contains the subject, the centred one often does not.
- Where the box is small (side 0-9 patches, 691 images, full FT) `cam` adds +2.46 vs +0.87
  for the centred box; on AdaptFormer, side 10-13 (13,034 images): +0.65 vs +0.03. Small,
  off-centre subjects are where location helps.
- Zooming to `mass` 0.5 drops context and hurts (zoom alone 64.9 / 73.3 AdaptFormer).
- 4 passes add only +0.04 to +0.13 over the 2-pass zoom.
- **CodaBench (2026-10-09), crop 0.75 + zoom, 2 passes, no retraining:** AdaptFormer final
  `af_final_45/last.pt` **0.797** (ID 972126; `scripts/predict.py --zoom`, commit d6d5800
  code; was 0.790 with crop 0.75 + flip). Anastasia's full FT at 336
  (`CS604_RUNS/full_ft_20261006_122039/last.pt`, team predictor `--zoom`, van-dev 8655284)
  **0.819** (ID 972132; was 0.813 with crop 0.7 + flip). +0.7 and +0.6, matching the holdout
  gains (+0.85 / +0.62). Re-predicting her crop 0.7 + flip on the laptop matched her Colab
  file on 49,945 of 50,000 rows (bf16, A100 vs RTX 5080).

## Task 2 AdaptFormer: learning-rate screen on the laptop (2026-10-07/08)

Plan and design: `report/ADAPTFORMER_PLAN.md`. Our own implementation (`mode: adaptformer`).
Common setup: 336 px, fixed 40/5 split (seed 604), all 10,000 classes, 1 epoch (1,563 steps),
width 192 in all 12 blocks (3,550,464 params = 4.13% of the 85,986,816-param pretrained
backbone, head excluded), scale 0.1, residual-stream input, Kaiming down / zero up, prototype
cosine head (prototypes_336) at head lr 4e-3, wd 0.05 (none on 1-D), label smoothing 0.1,
clip 1.0, batch 64 x 4 = 256, bf16, 5% warmup + cosine to zero. RTX 5080 laptop at 175 W.
Code at commit 61d8319 plus uncommitted AdaptFormer changes (`git_dirty: true` in run.json).

| Run | Adapter lr | Step 0 | Holdout top-1 | Holdout loss | Run time | Norm. 48 GB-h | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `af_a_lr3e-4` | 3e-4 | 42.56% | 70.564% | 1.403 | 36.2 min | 0.200 | 7.11 GB |
| `af_a_lr1e-3` | 1e-3 | 42.53% | **71.486%** | 1.345 | 36.4 min | 0.201 | 7.11 GB |
| `af_a_lr3e-3` | 3e-3 | 42.53% | **71.492%** | 1.331 | 36.1 min | 0.199 | 7.11 GB |
| `af_a_lr1e-2` | 1e-2 | 42.53% | 70.450% | 1.386 | 36.0 min | 0.199 | 7.11 GB |

- The step-zero gate passed on every arm (42.53% expected, tolerance 0.1 point): the adapters
  start as the identity. 42.56% vs 42.53% is bf16 nondeterminism.
- Flat top between 1e-3 and 3e-3 (3 images apart); both ends lose about 1 point. Both the
  middle values are inside the range, so no endpoint extension is needed.
- **Chosen: lr 1e-3** (tie rule: within 0.3 point, take the lower). 3e-3 had the lower holdout
  loss; recheck 1e-3 vs 3e-3 at 3 epochs in phase E if the budget allows.
- Throughput 216-218 img/s (full FT at 336: 167), so 1.3x faster than full FT.
- Phase B/C/D configs written against `af_a_lr1e-3`: `af_b_pca`, `af_b_norm`, `af_b_scale`,
  `af_c_ls0`, `af_c_margin01`, `af_d_w64`.

## Task 2 AdaptFormer: final run and CodaBench (2026-10-09)

`af_final_45` (commit c7bc90f, `notebooks/colab_train.ipynb`): the `af_e_combined_e3` recipe
on all 45 images/class, Colab A100-SXM4-40GB, `budget_48gb_hours: 2.0`. Stopped by the budget
after 8,755.6 s (2.00 normalized 48 GB-hours), 17,391 steps (just under 10 epochs), about
515 img/s, peak 14.53 GB. PCA init 10.2 s. Backbone trainable 3,550,476 = 4.13% (head
excluded); 11,230,477 = 11.55% with the head.

| Submission | Prediction | CodaBench val top-1 |
| --- | --- | --- |
| 971289 | plain, crop 0.9 | **0.779** |
| 971291 | crop 0.75 + flip (mean softmax, 2 passes) | **0.790** |

- Recorded with `scripts/record_submission.py` (`leaderboard.json`). The uploaded zips
  were renamed (`val_team_08_plaincrop0.9.zip`, `val_team_08_crop075.zip`); their SHA-256
  matches the zips `predict.py` wrote.
- Predicted on the laptop with `--skip-base-fingerprint`: the compact checkpoint's
  frozen-backbone fingerprint did not match, because the 336 px position embedding is
  resampled when the model is built and Colab's timm/torch resample it slightly
  differently. The checkpoint is insensitive to it: 98.80% / 98.82% / 98.64% on 5,000
  (seen) training images with the laptop's bicubic+antialias, bicubic without
  antialias, and bilinear embeddings. Both `.inference.json` files say
  `base_fingerprint_checked: false`.
- Crop 0.75 + flip added +1.1 on CodaBench (holdout: +0.87).
- Laptop prediction time (RTX 5080, 175 W, batch 128, includes loading): 2.88 ms/image
  plain, 4.64 ms/image with flip.
- Forward latency (`af_final_45/analysis/inference_latency.json`; the team launcher's method:
  synthetic 336 px input, bf16, 30 warm-ups, 200 timed iterations; RTX 5080 laptop at 175 W):

  | Model | Batch 1 | Batch 64 (per image) |
  | --- | --- | --- |
  | AdaptFormer | 5.60 ms | 1.83 ms |
  | AdaptFormer + flip (2 passes) | 10.58 ms | 3.70 ms |
  | Same ViT-B without adapters | 4.04 ms | 1.70 ms |

  The adapters add +1.56 ms (+39%) at batch 1 but only +7% at batch 64: at batch 1 the
  extra small kernels and hooks dominate. Unlike LoRA they cannot be merged into the
  weights. Teammates' latencies are on an A100, so compare within one GPU only.
- Comparison (CodaBench): AdaptFormer 0.779 / 0.790 at 4.13% of the backbone vs full FT at
  224 0.757 / 0.773 (ours), full FT at 336 0.807 / 0.813 (teammate, same recipe), last-block
  attention 0.668 (teammate, 2.75%), frozen probe 0.620-0.625; the course's PhD reference
  is 70.8% full FT and 68.3% LoRA (0.34%).

## Task 2 AdaptFormer: 3-epoch confirmation (2026-10-08)

Commit bd1bb63 (clean). Same setup as the screens below, 3 epochs (4,689 steps).
`af_e_control_e3` is `af_a_lr1e-3` for 3 epochs. `af_e_combined_e3` adds the four changes
that helped at 1 epoch: `adapter_input: norm`, `adapter_init: pca`,
`adapter_scale_trainable: true`, `cosine_margin: 0.1`.

| Epoch | Combined top-1 | Combined loss | Control top-1 | Control loss | Gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 42.53% | | 42.53% | | |
| 1 | 69.168% | 1.411 | 67.954% | 1.494 | +1.21 |
| 2 | 74.848% | 1.208 | 73.776% | 1.251 | +1.07 |
| 3 | **76.258%** | 1.226 | 75.270% | 1.230 | **+0.99** |

- **The combined recipe wins by about 1 point at every epoch** (tie threshold 0.4). It's
  the AdaptFormer recipe for the final run.
- Reference: full FT at 336 px, same 3 epochs: 78.39% (`full_proto_336_len3`). AdaptFormer is
  2.1 points behind it with 4.13% of the backbone trained.
- Run time: combined 113.8 min (0.629 normalized 48 GB-h; PCA calibration 49.7 s). Control
  131.1 min (0.725), slowed by the laptop dropping to a 55 W power limit at about step 3,850
  (throughput 212 -> 95 img/s). Accuracy is unaffected; only the control's time is.
- Learned branch scales in the combined run (blocks 1-12): 0.031, 0.031, 0.038, 0.043, 0.045,
  0.061, 0.069, 0.088, 0.072, 0.056, 0.044, 0.147. With the normalized input the scales
  shrank everywhere except the last block. As in `af_b_scale`, they rise with depth: the model
  wants more adaptation in deep blocks. Cosine temperature learned 20 -> 67.7.
- Combined-run PCA variance shares (normalized input, top 192 of 768): 0.99, 0.95, 0.81,
  0.77, 0.76, 0.74, 0.71, 0.71, 0.68, 0.64, 0.64, 0.73: less concentrated than the residual
  stream (75-100%).

Eval crop sweep on `af_e_combined_e3/best.pt` (`analysis/eval_tta.json`, commit bc3e3a3;
plain / + flip with mean softmax):

| crop_pct | 0.7 | 0.75 | 0.8 | **0.9** (training eval) |
| --- | --- | --- | --- | --- |
| plain | 76.732 | **76.952** | 76.802 | 76.258 |
| + flip | 76.922 | 77.126 | **77.192** | 76.526 |

- The 0.9 plain value reproduces the training run's 76.258% exactly (gate).
- Crop 0.75 and 0.8 tie with flip (77.13 / 77.19, 0.06 apart): about **+0.9 over plain
  0.9**. The full-FT sweep at 336 also peaked at 0.75. Choice for the submission:
  **crop 0.75 + flip** (consistent across both methods), two forward passes per image.
- All four crops with flips (8 passes): 78.228% (+1.97), at 8x the inference cost.
- `scripts/predict.py --crop-pct 0.75 --tta-flip` checked on val (50,000 rows validated,
  2 passes per image; 105 img/s on the laptop at a 55 W power limit).

## Task 2 AdaptFormer: initialization, loss and width screen (2026-10-08)

Same setup as the lr screen below, at lr 1e-3; each arm changes one setting from the control
`af_a_lr1e-3`. Code at commit cf70af0 (clean, except `af_d_w64`: `git_dirty` because the
untracked `af_c_margin02.yaml` was written during that run; the code was unchanged).

| Run | Change | Step 0 | Holdout top-1 | vs control | Holdout loss | Run time |
| --- | --- | --- | --- | --- | --- | --- |
| `af_a_lr1e-3` | control | 42.53% | 71.486% | | 1.345 | 36.4 min |
| `af_b_pca` | `adapter_init: pca` | 42.56% | 71.562% | +0.08 | 1.334 | 37.3 min |
| `af_b_norm` | `adapter_input: norm` | 42.56% | 71.722% | +0.24 | 1.322 | 36.5 min |
| `af_b_scale` | `adapter_scale_trainable: true` | 42.53% | 71.632% | +0.15 | 1.339 | 37.2 min |
| `af_c_ls0` | `label_smoothing: 0.0` | 42.56% | 71.164% | -0.32 | 1.210* | 36.8 min |
| `af_c_margin01` | `cosine_margin: 0.1` | 42.53% | **72.086%** | **+0.60** | 1.253 | 36.9 min |
| `af_d_w64` | `adapter_dim: 64` (1.38%) | 42.56% | 70.922% | -0.56 | 1.377 | 36.2 min |
| `af_c_margin02` | `cosine_margin: 0.2` | 42.56% | 70.944% | -0.54 | 1.598 | 37.3 min |

\* Not comparable: without smoothing the model is more confident, which lowers cross-entropy
at equal accuracy. Holdout loss is always plain cross-entropy (no smoothing, no margin).

- Every arm passed the step-zero gate, including PCA init (outputs unchanged by design).
- PCA calibration took 49.5 s of run time (16 batches of 64 at 336, plus starting the data
  workers). The top 192 of 768 directions held 100, 95, 88, 83, 79, 79, 80, 83, 80, 76, 75,
  79% of each block's input variance (blocks 1-12): the residual stream is very anisotropic.
- Learned scales (`af_b_scale`, blocks 1-12, all started at 0.1): 0.095, 0.075, 0.070, 0.082,
  0.083, 0.096, 0.122, 0.144, 0.162, 0.165, 0.150, 0.120. Early blocks shrank their branch and
  the later blocks grew it: the model wants more adaptation in deeper blocks.
- Only the margin clears the 0.4-point tie threshold. The three init/design changes are each
  small but all positive; label smoothing stays at 0.1; width 192 stays (64 loses 0.56).
- Margin 0.2 (`af_c_margin02`, commit 0f1f799, run after the others) lost 0.54: too strict at
  this temperature. Margin 0.1 stays.
- Next: 3-epoch confirmation, `af_e_control_e3` vs `af_e_combined_e3` (norm input, PCA init,
  learned scale, margin 0.1).

## Task 1 full fine-tune: LP-FT tuning on the laptop (2026-10-05)

**Why.** A teammate's full fine-tune (group repo branch `qq_branch`) reached
62.1% on CodaBench after 11 epochs at batch 32, with a constant learning rate
(no warmup or decay), a 100x head rate and no regularization. Plan: train a
linear probe on frozen features first, then fine-tune everything from that head
(LP-FT), with warmup and cosine decay, layer-wise LR decay and standard
regularization. Tune on the laptop; the final run goes on a Colab A100 40 GB
(2.4 h = 96 GB-hours). Backbone `vit_base_patch16_224.augreg2_in21k_ft_in1k`.

**Protocol.** Fixed split, 40 training / 5 holdout images per class
(`split_seed: 604`; holdout file identical to the prototype build). Scores are
internal holdout top-1 on 50,000 images (about +/-0.2 points standard error for
a single score). Laptop RTX 5080 Laptop GPU (15.92 GB, 175 W limit), so 1 h =
15.9 GB-hours.

**Stage 1: feature cache and linear probe** (commit `8b266e1`).
`feature_cache_augreg2.yaml` cached one eval-transform view per image: holdout
in 107 s, 400,000 training images in 412 s (970 img/s). The cached holdout
reproduced the prototype score exactly: 43.900%. The first attempt hit a CUDA
out-of-memory error at about 51k training images (peak use on the retry was
2.7 GB of 16 GB, so most likely a transient spike from another process); log
kept as `logs/feature_cache_augreg2_attempt1_oom.log`.
`head_sweep_augreg2.yaml`: cosine head from the prototypes, trained temperature,
batch 256, 10 epochs (15,630 updates), about 36 s per point:

| Point | Head lr | Label smoothing | Holdout top-1 | Final temperature |
| --- | --- | --- | --- | --- |
| r1_lr1e-3_ls01 | 1e-3 | 0.1 | 60.354% | 64.9 |
| r1_lr3e-3_ls01 | 3e-3 | 0.1 | 60.370% | 65.1 |
| r1_lr1e-2_ls01 | 1e-2 | 0.1 | 60.324% | 65.0 |
| r1_lr3e-2_ls01 | 3e-2 | 0.1 | 60.394% | 64.7 |
| r1_lr3e-3_ls0 | 3e-3 | 0 | 61.064% | 101.4 |
| r1_lr1e-2_ls0 | 1e-2 | 0 | 60.888% | 105.4 |

The head's rate hardly matters. Without label smoothing the head scores about
0.7 points higher, but its temperature passes the head's clamp at 100. The
fine-tune starts from `r1_lr3e-3_ls01`, trained with the fine-tune's own loss
(label smoothing 0.1). A head alone on frozen features (60.4%) is within 2
points of the teammate's full fine-tune.

**Stage 2, Round A: backbone learning rate** (`full_lpft_a_lr*.yaml`). One
epoch (1,563 updates of batch 64 x 4 = 256), head from `r1_lr3e-3_ls01`,
step-zero gate 60.37% (reproduced exactly in every run). Recipe: AdamW,
`layer_decay: 0.75`, weight decay 0.05 with none on norms/biases/tokens, label
smoothing 0.1, drop_path 0.1, gradient clip 1.0, bf16, 5% warmup then cosine to
zero; the head shares the top rate. Each run: about 1,320 s, 0.122 normalized
48 GB-hours, 5.82 GB peak allocated. Parameters: 93,478,657 total; backbone
85,798,656 trainable; cosine head 7,680,001.

| Run | Commit | Backbone lr | Holdout top-1 | Holdout loss |
| --- | --- | --- | --- | --- |
| full_lpft_a_lr5e-5 | 343bb3c | 5e-5 | 66.236% | 1.658 |
| full_lpft_a_lr1e-4 | 343bb3c | 1e-4 | 66.820% | 1.621 |
| full_lpft_a_lr2e-4 | 343bb3c | 2e-4 | 67.220% | 1.586 |
| full_lpft_a_lr4e-4 | abbf48d | 4e-4 | **67.474%** | 1.553 |
| full_lpft_a_lr8e-4 | e7467f0 | 8e-4 | 67.340% | 1.538 |

The optimum is flat between 4e-4 and 8e-4. At 8e-4 the training loss rose
early (to 3.58 near step 350, from about 3.0) before recovering. Round B uses
4e-4, since longer runs usually favour a slightly lower rate. One epoch of LP-FT
on the laptop (0.12 normalized hours) already scores 67.5% on the internal
holdout.

**Stage 2, Round B: head rate and layer decay at 3 epochs**
(`full_lpft_b_*.yaml`, 4,689 updates, backbone lr 4e-4). Each run: about
3,820 s, 0.35 normalized 48 GB-hours, 5.82 GB peak. Holdout top-1 after each
epoch (the schedule decays to zero at the end of epoch 3):

| Run | Commit | Change from control | Epoch 1 | Epoch 2 | Epoch 3 | Final loss |
| --- | --- | --- | --- | --- | --- | --- |
| full_lpft_b_control | dc244e0 | none (head shares 4e-4) | 63.858% | 68.566% | 69.832% | 1.505 |
| full_lpft_b_headlr10x | aac0bf2 | head lr 4e-3 | 62.182% | 69.196% | **71.366%** | 1.400 |
| full_lpft_b_nolayerdecay | aac0bf2 | `layer_decay: null` | 44.896% | 58.444% | 63.120% | 1.778 |

- Three epochs instead of one: +2.36 points (67.474% to 69.832%).
- A head rate 10x the backbone's: +1.53 points, beyond holdout noise.
- Without layer-wise decay, 4e-4 on every layer drops below the 60.37% start
  in epoch 1 and ends 6.7 points lower: the early layers' pretrained features
  are damaged. Layer decay stays.

**Stage 2, Round C: one change each from `full_lpft_b_headlr10x`**
(`full_lpft_c_*.yaml`, 3 epochs, about 3,800 s and 0.35 normalized hours each):

| Run | Commit | Change from head-lr-10x | Epoch 1 | Epoch 2 | Epoch 3 | Final loss |
| --- | --- | --- | --- | --- | --- | --- |
| full_lpft_b_headlr10x (reference) | aac0bf2 | none | 62.182% | 69.196% | 71.366% | 1.400 |
| full_lpft_c_headlr30x | c5672a5 | head lr 1.2e-2 (30x) | 58.484% | 68.802% | **71.966%** | 1.299 |
| full_lpft_c_crop035 | 6c90bcc | `train_crop_scale: [0.35, 1.0]` | 62.248% | 69.144% | 71.106% | 1.430 |
| full_lpft_c_nojitter | 6c90bcc | `color_jitter: 0.0` | 62.250% | 69.394% | 71.286% | 1.407 |

- Crop scale and colour jitter make no difference at 3 epochs (within about
  0.3 points); keep timm's defaults.
- Head lr 30x: +0.60 points and lower loss, but it starts lower (58.5% after
  epoch 1). Seed repeats of both head rates decide whether the gap is real.
- `full_lpft_c_headlr30x` stopped at the start of epoch 3 when a data-loader
  worker failed to read
  `train/class_1986/47ca8fb4-65e5-4041-b5ed-29f90997a2f6.jpg`
  (`OSError: [Errno 22] Invalid argument`). The file opens and decodes
  normally, so this was a transient Windows read error. The run resumed from
  `last.pt` (end of epoch 2, step 3,126, `resumed_at_elapsed_sec` 2,577) at
  commit c5672a5; `metrics.csv` repeats a few epoch-3 training rows from the
  first attempt. Logs: `logs/full_lpft_c_headlr30x.log` and
  `logs/full_lpft_c_headlr30x_resume.log`.

**Stage 3: seed repeat and the head-initialization ablation** (3 epochs, head
lr 4e-3, otherwise the Round B recipe):

| Run | Commit | Head start (step-zero top-1) | Seed | Epoch 1 | Epoch 2 | Epoch 3 | Final loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_lpft_b_headlr10x | aac0bf2 | linear probe (60.37%) | 0 | 62.182% | 69.196% | 71.366% | 1.400 |
| full_lpft_b_headlr10x_seed1 | eb37388* | linear probe (60.37%) | 1 | 61.724% | 69.378% | 71.034% | 1.409 |
| full_proto_headlr10x | 4ed50cd | prototypes (43.926%) | 0 | 60.580% | 70.172% | **72.984%** | 1.259 |

\* `git_dirty: true`: only `report/EXPERIMENTS.md` and the Colab notebook were
modified when it started; training code as committed.

- Seed noise: the two seeds of one recipe differ by 0.33 points.
- **Starting from the prototypes beats starting from the linear probe by 1.6
  to 2.0 points**, far beyond seed noise, although the linear-probe start is
  16 points better at step zero. LP-FT does not help here. A plausible reason:
  the probe head is already confident (temperature about 65 with label
  smoothing), so early gradients into the backbone are small and the features
  adapt less; the prototype head starts at temperature 20. LP-FT's usual
  benefit is preserving features for out-of-distribution robustness, which this
  in-distribution test does not reward.
- Consequence: the remaining runs and the final run start from the prototypes.
  The not-yet-run `full_lpft_len6` and `full_lpft_c_headlr30x_seed1` configs
  were removed (commit ad81743) and replaced by `full_proto_headlr30x`.

**Stage 4: head rate from the prototypes, and the length curve.**

| Run | Commit | Head lr | Epochs | Per-epoch holdout top-1 | Final loss | Normalized 48 GB-h |
| --- | --- | --- | --- | --- | --- | --- |
| full_proto_headlr10x | 4ed50cd | 4e-3 | 3 | 60.580, 70.172, **72.984%** | 1.259 | 0.344 |
| full_proto_headlr30x | ad81743 | 1.2e-2 | 3 | 57.634, 68.764, 72.706% | 1.257 | 0.345 |
| full_proto_len6 | ea0231e | 4e-3 | 6 | 58.114, 64.320, 68.268, 71.472, 73.658, **73.966%** | 1.302 | 0.690 |
| full_proto_len12 | dac0784 | 4e-3 | 12 | 56.276, 62.282, 64.652, 66.440, 68.148, 69.646, 70.850, 72.134, 73.254, 74.090, 74.404, **74.494%** | 1.410 | 1.380 |

- From the prototypes the head rate no longer matters (30x is 0.28 points
  lower, within seed noise); the recipe keeps 10x.
- Doubling the length from 3 to 6 epochs adds +0.98 points (72.984% to
  73.966%; 7,489 s on the laptop). Holdout loss rose slightly in the last epoch
  (1.285 to 1.302) while accuracy still rose: gains are slowing, not reversed.
- 12 epochs (2026-10-06, 14,982 s on the laptop; the first launch crashed
  when a second process exhausted Windows virtual memory, kept as
  `full_proto_len12_crash_vmem`): **74.494%**, only +0.53 over 6 epochs for
  twice the compute. Holdout loss bottomed at epoch 8 (1.377) and rose to 1.410
  while accuracy crept up; the model grows more confident, not less accurate.
  For the final run (~16 epochs on 45 images/class) length is close to
  saturated; more regularization is the remaining lever on this axis.

**Chosen recipe and final run.** `full_proto_final_45.yaml`: the
`full_proto_headlr10x` recipe on all 45 images/class (`val_per_class: 0`, no
holdout, predict from `last.pt`), effective batch 256 (256 x 1 on the A100),
`budget_48gb_hours: 2.0` (about 2.4 h on a 40 GB A100; the cosine schedule ends
on the clock), `epochs: 30` as an upper bound only. The A100's full fine-tune
throughput is not measured yet; at 600 to 900 img/s the budget allows about
12 to 17 epochs, beyond the 6 tested here.

Laptop tuning total for this section: 18 training runs plus the cache and
probe, about 17.5 laptop hours.

**Stage 5 (2026-10-06): evaluation-only crop and flip, no training.**
`scripts/eval_tta.py` (commit `dac0784`) re-scores a checkpoint on its own
holdout list with a different eval center-crop fraction (`crop_pct`; the
backbone's default is 0.9, which training's evaluation used), and with the
softmax averaged over the image and its horizontal flip. The default crop
reproduced each run's final holdout score (checked by hand: exact for 3, 6 and
336 px runs, 0.02 points off at 12 epochs; the script's automatic gate read the
wrong `metrics.csv` column until commit `459fe70` and silently skipped).
Each crop takes about
100 s on the laptop (two forward passes per image). Results in
`cs604-outputs/<run>/analysis/eval_tta*.json`.

| crop_pct | 6 epochs plain | 6 epochs + flip | 3 epochs plain | 3 epochs + flip |
| --- | --- | --- | --- | --- |
| 1.0 | 72.766% | 73.134% | | |
| 0.95 | 73.550% | 73.866% | | |
| **0.9 (default)** | **73.966%** | 74.342% | **72.984%** | 73.358% |
| 0.875 | 74.418% | 74.678% | | |
| 0.85 | 74.596% | 74.998% | | |
| 0.8 | 75.036% | 75.444% | | |
| 0.75 | 75.338% | 75.542% | 74.306% | 74.546% |
| 0.7 | 75.380% | 75.504% | 74.568% | **74.854%** |
| 0.65 | 75.502% | **75.722%** | 74.442% | 74.808% |
| 0.6 | 75.142% | 75.392% | | |

12 epochs (`full_proto_len12/best.pt`), plain / + flip: crop 0.9 74.474 / 74.738%
(gate: 74.494%, within 0.02 points); 0.8 75.438 / 75.620%; 0.75 75.512 / 75.780%;
**0.7 75.704 / 76.000%**; 0.65 75.844 / 76.014%; 0.6 75.570 / 75.726%. The
plain crop gain shrinks slowly with length (0.9 to 0.7: +1.58, +1.41, +1.23
points at 3, 6, 12 epochs) but crop 0.7 + flip still adds **+1.51** at 12
epochs, and the optimum stays at 0.65 to 0.7. Averaging the six crops with
flips: 77.430%.

**Confirmed on CodaBench (2026-10-06).** The final A100 run of the group
repo's `configs/full_ft_224px_proto-cosine_llrd0.75_top-lr4e-4_2.4h.yaml` (all 45 images/class, 144 training
minutes): plain center crop 0.9 **0.757** (ID 963899); the same `last.pt`
re-predicted with `--crop_pct 0.7 --tta_flip` **0.773** (ID 964799), +1.6
points, in line with the holdout estimate (+1.51 at 12 epochs).

**Stage 6 (2026-10-07): the recipe at 336 px.** A teammate's A100 run of the
group config at 336 px (`full_ft_336px_proto-cosine_llrd0.75_top-lr4e-4_2.4h.yaml`, ~8.5 epochs in 144 min) scored
0.807 plain / 0.813 with crop 0.7 + flip on CodaBench, +5.0 over 224, but the
crop gain was only +0.6 and that run had no holdout. `full_proto_336_len3`
(commit `5661383`): the `full_proto_headlr10x` recipe with `image_size: 336`,
prototypes rebuilt at 336 (`prototype_init_336.yaml`: **42.534%**, vs 43.90% at
224; frozen features do not gain from resolution), batch 32 x 8 = 256.
169 img/s, 6.31 GB peak, 7,950 s, 0.732 normalized 48 GB-hours.

| Run | Epoch 1 | Epoch 2 | Epoch 3 | Final loss |
| --- | --- | --- | --- | --- |
| full_proto_headlr10x (224 px) | 60.580% | 70.172% | 72.984% | 1.259 |
| full_proto_336_len3 (336 px) | 65.850% | 75.608% | **78.388%** | 1.047 |

- **+5.40 points at the same 3 epochs** (2.3x the compute per epoch), matching
  the teammate's +5.0 on CodaBench. 3 epochs at 336 already beat 12 epochs at
  224 (74.494%) with less compute (0.73 vs 1.38 normalized hours).

Eval crop sweep at 336 px (`full_proto_336_len3/analysis/eval_tta.json`), plain / + flip:

| crop_pct | 0.6 | 0.65 | 0.7 | 0.75 | 0.8 | 0.85 | **0.9** | 1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | 77.860 | 78.314 | 78.636 | **78.798** | 78.696 | 78.628 | 78.388 | 77.646 |
| + flip | 78.112 | 78.544 | 78.762 | **78.970** | 78.848 | 78.796 | 78.514 | 77.948 |

- At 336 the crop effect is much smaller: best crop **0.75**, +0.41 plain and
  **+0.58 with flip** over the default; crop 0.7 + flip +0.37. This matches the
  teammate's +0.6 on CodaBench. Recommendation at 336: crop 0.75 + flip
  (expected only about +0.2 over 0.7 + flip).
- Averaging all eight crops with flips: 80.202% (+1.8), but 16 forward passes.

6 epochs = `full_proto_len6/best.pt`, 3 epochs = `full_proto_headlr10x/best.pt`;
"+ flip" is the mean softmax (mean logits is within 0.05 points). Holdout loss
falls the same way (6 epochs: 1.302 at 0.9, 1.220 at 0.65).

- A tighter test crop is worth +1.4 to +1.6 points, flat between 0.65 and
  0.75, at both lengths. Likely the train/test object-size mismatch (FixRes):
  timm's RandomResizedCrop (scale 0.08 to 1) shows objects larger in training
  than the 0.9 center crop does at test time.
- Flip averaging adds +0.1 to +0.4 on top (smaller at the best crop).
- Chosen: **crop 0.7 + flip**, +1.54 (6 epochs) and +1.87 (3 epochs). The
  optimum did not move between 3 and 6 epochs; the 12-epoch run checks it
  again: it holds (+1.51 at 12 epochs, below).
- Averaging all four crops 0.875-1.0 with flips: 75.142%; crops 0.6-0.7 with
  flips: 76.846% (6 epochs, multi-crop ensembles cost 6-8 forward passes).
  Not adopted: inference latency is reported, and 0.7 + flip gets most of it.
- Group repo (`van-dev`, commits `f01bcde`, `575ee88`, not pushed): `src.predict`
  gets opt-in `--crop_pct` and `--tta_flip` (default output byte-identical to
  before), and `colab_van_train-one-config.ipynb` writes a second submission with crop
  0.7 + flip to `tta_crop070_flip/`.

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
