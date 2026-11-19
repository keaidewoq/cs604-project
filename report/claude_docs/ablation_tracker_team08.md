<!-- Snapshot of the shared Claude doc https://claude.ai/artifact/F5K91oPAL2sn4vzmvaekeB at revision 192, taken 2026-11-11. Read-only copy for version history: edit the live doc, not this file. Images and embedded diagrams are not included; open suggestions are left out of the text and listed at the end. -->

# CS604 Ablation Tracker (Team 08)

2026-11-10 · @member

Every ablation in the final report draft, split by full fine-tuning and PEFT, so each number can be traced to a run record before 18 Nov. Numbers are copied from the report as of today; holdout = Van's 10,000-class holdout unless stated.

**How to update:** set **Status** for your rows, put your name in **Checked by**, and add the run name or file in **Run record**. Add a row for any ablation that is missing. Use **Conflict** when the run record disagrees with the report, and note both numbers.

## Final full fine-tuning config: what justifies each choice

Working backwards from the 84.07% model. Four settings rest on a direct ablation, two on partly recorded evidence (top LR, schedule), and the rest, including layer decay, on the literature or the brief.

| Setting | Final value | Justified by | Evidence |
| --- | --- | --- | --- |
| Base model | ViT-B/16 augreg2 (in21k → in1k) | Course brief | Required, not chosen |
| Head | Prototype-initialised cosine head, learned temperature | F3: beat LP-FT by 1.61 pp | Direct ablation (one seed, 224 px, 3 ep) |
| Head LR | 10× the backbone (4e-3) | F1: +1.53 pp over 1×; F12: 30× tied | Direct ablation |
| Resolution | 448 px | F7: 336 px +5.4 pp over 224; F8: 512 px only +0.19 over 448 at equal compute; F9, F10: low-res shortcuts lost | Direct ablation |
| Epochs | 5.5 (whatever fits 144 min at 448 px) | F6: 6 → 12 epochs only +0.53 pp, so compute went to pixels | Direct ablation (224 px) |
| Layer decay | 0.75 per block | literature review | - |
| Top LR | 4e-4 | F4: five-point sweep, decay on | Numbers only in Appendix A |
| Schedule | 5% warm-up, cosine decay fitted to the clock | F13: decay to zero beat constant LR by 9.0 at 3 ep (controlled) | Measured: 72.98 vs 63.99 on holdout, one seed |
| Augmentation | Default crop scale and colour jitter | F12: alternatives tied | Tie, numbers not listed |
| Label smoothing 0.1, drop path 0.1, weight decay 0.05, clipping 1.0 | As listed | MAE / DeiT / ViT recipes | Literature only, not ablated |
| Batch 256, bf16 | As listed | Fits 48 GB | Practical choice |
| Prediction | Aimed zoom + kNN (mean of 3 closest, K = 5) | X1, X2, X4, X5 | Direct ablation, held on Phase 2 (X6) |

## Full fine-tuning

Recipe ablations ran on the holdout at 224 px, 3 epochs, one seed, unless the Setup column says otherwise (report Sections 3.3, 3.4, 5.2; Table 3; Figures 4 and 5).

| # | Ablation | Setup | Result (top-1 %) | Contribution to final FT | Status | Checked by | Run record |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | Head LR 10× the backbone vs head at backbone LR | Holdout, 224 px, 3 ep | 69.83 → 71.37 (+1.53) | Kept: head LR 10× the backbone (4e-3) | Verified | Van | full_lpft_b_control 69.832 vs full_lpft_b_headlr10x 71.366 (epoch 3). Van's EXPERIMENTS.md, LP-FT tuning, Round B |
| F2 | No layer decay, same top LR (one-sided: LR not re-tuned) | Holdout, 224 px, 3 ep | 69.83 → 63.12 (−6.71) | None: not used as justification (LR not re-tuned without decay). Layer decay is justified by the literature | Verified | Van | full_lpft_b_nolayerdecay 63.120 vs full_lpft_b_control 69.832. LP-FT tuning, Round B |
| F3 | Prototype head vs LP-FT | Holdout, 224 px, 3 ep | 71.37 → 72.98 (+1.61) | Kept: prototype cosine head replaced LP-FT | Verified | Van | full_proto_headlr10x 72.984 vs full_lpft_b_headlr10x 71.366 (seed 0). LP-FT tuning, Stage 3 |
| F4 | Top LR, five-point sweep (decay on) | Appendix A | Not in the report body | Set the top LR to 4e-4 | Verified | Van | full_lpft_a_lr5e-5 / 1e-4 / 2e-4 / 4e-4 / 8e-4: 66.24 / 66.82 / 67.22 / 67.47 / 67.34 (1 epoch, layer decay on, LP-FT head). LP-FT tuning, Round A. Record found |
| F5 | Head initialisation, linear head only (frozen backbone) | Not in the report | Anastasia recalls running it | None yet. Would back the prototype head from a second angle | Missing record |  |  |
| F6 | 6 / 12 epochs vs 3 | Holdout, 224 px | 73.97 / 74.49 (+0.98, then +0.53 more) | Ruled out spending compute on epochs; budget went to resolution instead | Verified | Van | full_proto_headlr10x 72.98 / full_proto_len6 73.97 / full_proto_len12 74.49. LP-FT tuning, Stage 4 |
| F7 | 336 px vs 224 px | Holdout, 3 ep | 78.39 (+5.4) | Showed resolution is the biggest lever; moved training above 224 px | Verified | Van | full_proto_336_len3 78.388 vs full_proto_headlr10x 72.984. LP-FT tuning, Stage 6 |
| F8 | Resolution at equal compute | Holdout, 10 img/class, same GPU budget | 63.20 / 64.06 / 64.79 / 64.98 at 336 / 384 / 448 / 512 px | Chose 448 px: 512 px added only 0.19 | Verified | Van | res_screen_336 / 384 / 448 / 512, crop 0.9, 0.48 normalised 48 GB-h each. Full FT training resolution at equal compute (2026-10-15/17) |
| F9 | Progressive 224 → 384 px vs plain 384 px | Holdout, 2.5× the epochs | −4.3 | Ruled out progressive resizing; train at 448 px from the start | Verified | Van | res_screen_224to384 59.74 vs res_screen_384 64.06 (crop 0.9), −4.32; same 0.48 GB-h budget |
| F10 | 224 px + short 336 px finish vs plain 336 px | Holdout | −1.85 | Ruled out a FixRes-style high-res finish | Verified |  |  |
| F11 | Frozen prototype accuracy by resolution | Holdout, no training | 43.9 (224 px) vs 37.2 (448 px) | No config change. Explains why high resolution needs the backbone to adapt | Verified | Van | prototype_init (224) 43.90; prototype_init_336 / 384 / 448 / 512: 42.53 / 40.47 / 37.20 / 33.85 |
| F12 | Crop scale, colour jitter, 30× head LR | Holdout, Figure 4 | Ties; numbers not listed | Kept the defaults and 10× head LR (alternatives tied) | Verified | Van | From the LP-FT head: full_lpft_c_crop035 71.11, full_lpft_c_nojitter 71.29 vs 71.37. Head LR 30x from prototypes: full_proto_headlr30x 72.71 vs 72.98 (from the LP-FT head 71.97 vs 71.37). LP-FT tuning, Rounds C and Stage 4 |
| F13 | Warm-up + cosine decay to zero vs warm-up + constant LR (schedule the only change) | Van holdout (all 10k classes, 40/5 split), full FT, 224 px, 3 ep, seed 0; constant arm holds 4e-4 top / 4e-3 head after the same 5% warm-up | 72.98 vs 63.99, decay +8.99 (per epoch 60.58 / 70.17 / 72.98 vs 58.17 / 62.35 / 63.99). Clear win, one seed | Kept: warm-up + cosine decay to zero, now backed by a controlled run. A lower constant LR was not tried | Verified | Van | Controlled run 2026-11-11: `full_proto_constlr` vs `full_proto_headlr10x` (EXPERIMENTS.md, RERUNS.md item 4). Replaces the old 1 ep vs 11 ep comparison (67.47 holdout vs 0.621 CodaBench, different recipes). Suggest Status = Verified |
| F14 | CodaBench path: recipe, then resolution | CodaBench Phase 1, runs not perfectly matched | 66.3 → 75.7 → 80.7 / 81.7 / 82.5 (336 / 384 / 448 px) | Measures the total: recipe +9.4, resolution +6.8 on CodaBench | To verify | Qian | Qian — 66.3% baseline, ID 962394 |
| F15 | Head trained alone on frozen features (prototype-initialised cosine head) | Holdout, 224 px, cached frozen features, 10 ep | 43.90 → 60.37. Head lr 1e-3 to 3e-2 flat (60.32–60.39); no label smoothing 61.06 but temperature passes the clamp at 100 | None: the start head of the LP-FT arms (F1–F3) only. Related to F5 | Newly added, verified | Van | feature_cache_augreg2 + head_sweep_augreg2 (points r1_*). Van's EXPERIMENTS.md, LP-FT tuning, Stage 1 |
| F16 | Seed repeat of one recipe | Holdout, 224 px, 3 ep, LP-FT head, seeds 0 and 1 | 71.37 vs 71.03 (0.33 apart) | Sets the seed noise behind the 0.4-point tie rule | Newly added, verified | Van | full_lpft_b_headlr10x vs full_lpft_b_headlr10x_seed1. LP-FT tuning, Stage 3 |
| F17 | Base-model screen: class-prototype accuracy of four ViT-B checkpoints | Holdout, 224 px, no training, 40-image prototypes | orig_in21k 41.82; augreg2_in21k_ft_in1k 43.90; augreg_in21k 50.01; DINOv2 ViT-B/14 53.76 | None: team locked augreg2 (2026-10-05). Report comparison only | Newly added, verified | Van | prototype_validate_<variant> (baseline.json). Van's EXPERIMENTS.md, Base-model screen |
| F18 | Frozen-head ablation: LoRA r8 qkv with the prototype head never trained | Holdout, 224 px, head frozen at the prototypes | 43.90 → 51.09 (10/class, 2 ep), 51.83 (40/class, 1 ep), 52.92 (budget run, 6.7 ep). Head-only, backbone frozen, 5/class 1 ep: 45.86 | None: superseded by the Prof's rule that the head trains (2026-10-03) | Newly added, verified | Van | prototype-lora-r8-fp32-screen, prototype-lora-r8-fp32-40, prototype_lora_r8_fp32_budget_40, prototype-head-screen. Van's progress_log 2026-09-30 to 2026-10-03 |
| F19 | Neighbour label smoothing (0.1 mass onto the ±5 neighbouring class IDs) vs uniform | Holdout, 384 px, 10 img/class, 0.48 normalised GB-h each | 63.92 vs 64.06 at crop 0.9 (−0.14, tie) | Dropped; uniform smoothing 0.1 kept | Newly added, verified | Van | nbr_smooth_384_w5 vs res_screen_384. Van's EXPERIMENTS.md, neighbour label smoothing (2026-10-16) |

## PEFT

From report Sections 3.5, 5.3, 5.4 and 6, Table 4 and Figure 5. Holdout runs are at 336 px, 3 epochs unless stated.

### All methods compared

| # | Ablation | Setup | Result (top-1 %) | Status | Checked by | Run record |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | Method comparison: 9 methods (Table 4). To be replaced by the controlled comparison | CodaBench Phase 1, 336 px, 144 min; recipes and batch sizes differ | 67.93 (last-block attention) to 78.5 (LoRA r24 + extras); top six tie at 77.6–78.5 | To verify |  |  |
| P21 | SSF alone (scale and shift after every op, 0.24% of backbone) vs the AdaptFormer recipe (4.27%) | Van holdout, 336 px, 1 ep, lr 1e-3 / 3e-3 / 1e-2 | Best 71.90 at lr 3e-3 vs 72.97 (−1.06); 1e-3 71.56, 1e-2 70.69. 15% slower than AdaptFormer and +5.3 GB. Not adopted |  | Van | ssf_lr1e-3_e1, ssf_lr3e-3_e1, ssf_lr1e-2_e1 vs af_bn_bias_norm_e1. Van's EXPERIMENTS.md, SSF first screen (2026-10-20) |

### LoRA

| # | Ablation | Setup | Result (top-1 %) | Status | Checked by | Run record |
| --- | --- | --- | --- | --- | --- | --- |
| P2 | AdaLoRA vs LoRA | 1,000-class pilot | −2.7; not run at full scale | Verified | Qian | The [2026-10-23 entry (line 589)](report/progress_log.md:589) is headed **“Qian — LoRA vs AdaLoRA.”** LoRA **90.76%**; AdaLoRA targets 8/16/20: **87.92 / 87.94 / 88.04%**. Best AdaLoRA is **−2.72 percentage points****.** |
| P4a | LoRA placement: all 12 blocks vs last 8, equal budget | Holdout, two seeds | 77.26 vs 77.33 (tie) | Verified | Van | eqp_lora42_b4_e3_s0/s1 77.33 / 77.32 vs eqp_lora28_e3_s0/s1 77.23 / 77.28 (+0.07, tie); 4.94% each. Van's RERUNS.md item 3 |
| P7 | LoRA alone vs AdaptFormer + LoRA, same budget | Holdout | 77.26 vs 77.43 (tie) | Verified | Van | eqp_lora28_e3_s0/s1 77.23 / 77.28 vs eqp_af_lora16_e3_s0/s1 77.44 / 77.41 (+0.17, tie); 4.94% vs 4.96%. RERUNS.md item 1 |
| P13 | LoRA vs full FT by scale | 1,000-class pilot vs full scale, 336 px | Pilot: LoRA matched or beat full FT. Full scale: full FT +2.2 | To verify |  |  |
| P14 | LoRA rank: r16 vs r24, all 12 blocks (arm C of the r24 recipe ablation) | 1,000-class pilot, 336 px, 3 epochs, one seed; repeated in the depth-control screen (C16) | 89.84 (r24, 4.65%) vs 89.18 (r16, 3.28%): r24 +0.66. Repeat: r16 89.32, −0.54. Report cites +0.66 | To verify |  | configs/lora_r24_336px_recipe-ablation_1k-class-pilot.yaml (CS604_RUNS/ablation_ablation_lora_r24/, arm C); configs/lora_r24_336px_spread-vs-fewer-blocks_1k-class-pilot.yaml (C16); progress_log 2026-10-12 and 2026-10-19 |

### AdaptFormer

| # | Ablation | Setup | Result (top-1 %) | Status | Checked by | Run record |
| --- | --- | --- | --- | --- | --- | --- |
| P3 | Add biases + LayerNorms to AdaptFormer | Holdout | 76.26 → 76.76 (+0.50) for 0.14% of backbone | To verify | Van | af_bn_bias_norm_e3 76.756 vs af_e_combined_e3 76.258 (3 ep, 336 px); 1 ep: biases alone +0.44, + LN weights +0.57. CodaBench 0.802 vs 0.797 (IDs 980309 / 972126) |
| P4b | AdaptFormer placement: all 12 blocks vs last 8, equal budget | Holdout, two seeds | 76.77 vs 76.90 (tie) | To verify | Van | eqp_af288_b4_e3_s0/s1 76.97 / 76.83 vs eqp_af_bn_e3 76.77 / 76.77 (+0.13, tie); 4.27% each. RERUNS.md item 3 |
| P5 | Wider adapters (4.27% → 4.96%) | Holdout | 76.77 → 76.84 (no gain) | To verify | Van | eqp_af224_e3_s0/s1 76.81 / 76.87 (mean 76.84) vs eqp_af_bn_e3 76.77. RERUNS.md item 1 |
| P11 | AdaptFormer resolution at equal compute | Holdout, 10 img/class (Figure 5) | Keeps rising to 448 px; 512 px not run | To verify | Van | af_res_screen_336 / 384 / 448 at crop 0.9: 59.63 / 61.18 / 62.72 (+1.55, +1.54), 0.48 GB-h each. CodaBench plain 384 0.794, 448 0.804 |
| P15 | Adapter learning rate | Van holdout, 336 px, 1 ep, width 192 | 3e-4 70.56, 1e-3 71.49, 3e-3 71.49, 1e-2 70.45. Flat top; 1e-3 kept (tie rule: lower rate) |  | Van | af_a_lr3e-4, af_a_lr1e-3, af_a_lr3e-3, af_a_lr1e-2. Van's EXPERIMENTS.md, AdaptFormer learning-rate screen |
| P16 | Adapter width 64 (1.38%) vs 192 (4.13%) | Van holdout, 336 px, 1 ep | 70.92 vs 71.49 (−0.56). Width 192 kept |  | Van | af_d_w64 vs af_a_lr1e-3. AdaptFormer initialisation, loss and width screen |
| P17 | Loss settings: CosFace margin 0.2, no label smoothing | Van holdout, 336 px, 1 ep, vs control 71.49 | Margin 0.2: 70.94 (−0.54; margin 0.1 +0.60). Label smoothing 0: 71.16 (−0.32). Margin 0.1 and smoothing 0.1 kept |  | Van | af_c_margin02, af_c_ls0, af_c_margin01. AdaptFormer initialisation, loss and width screen |
| P18 | Regularisation: RandAugment m9 + random erasing 0.25; adapter dropout 0.1 | Van holdout, 336 px, 3 ep, vs 76.76 | RandAugment 76.44 (−0.32), adapter dropout 76.72 (−0.03). Both ties; not adopted |  | Van | af_aug_ra_re_e3, af_adrop01_e3 vs af_bn_bias_norm_e3. AdaptFormer regularisation screen (2026-10-17/18) |
| P19 | Depth-graded adapter widths 96 / 192 / 288 by block third, same 4.27% | Van holdout, 336 px, 1 ep | 72.89 vs 72.97 (−0.07, tie). Uniform width 192 kept |  | Van | af_width_graded_e1 vs af_bn_bias_norm_e1 (2026-10-18) |
| P20 | Learning rate for the trainable biases/norms | Van holdout, 336 px, 1 ep, vs 72.97 | Biases/norms at 3e-3: 72.66 (−0.31); everything at 3e-3: 72.86 (−0.11). Ties; one lr 1e-3 kept |  | Van | af_bn_biaslr3e-3_e1, af_bn_lr3e-3_e1 (2026-10-11) |
| P22 | SSF beside AdaptFormer in place of trainable biases/norms | Van holdout, 336 px, 1 ep | 73.18 vs 72.97 (+0.21, tie), but 21% slower and +6.4 GB. Biases/norms kept |  | Van | af_ssf_e1 vs af_bn_bias_norm_e1 (2026-10-20) |

### Final design: AdaptFormer + LoRA

| # | Ablation | Setup | Result (top-1 %) | Status | Checked by | Run record |
| --- | --- | --- | --- | --- | --- | --- |
| P6 | Add LoRA to AdaptFormer instead of widening, same budget | Holdout, two seeds | 77.43 (+0.6 in both seeds) | To verify | Van | eqp_af_lora16_e3 77.44 / 77.41 vs eqp_af224_e3 76.81 / 76.87: +0.63 / +0.55 at equal 4.96%. RERUNS.md item 1 |
| P8 | CosFace margin 0.1 + three small design changes | Holdout | +0.99 from 75.27; the three changes are not named in the report | To verify | Van | af_e_combined_e3 76.258 vs af_e_control_e3 75.270 (3 ep). Changes: norm2 input +0.24, PCA init +0.08, learned scale +0.15, margin 0.1 +0.60 (each at 1 ep: af_b_norm, af_b_pca, af_b_scale, af_c_margin01) |
| P9 | Final build path on the holdout | Holdout | Slides (two-seed mean): 77.43, LoRA +0.66. Van's draft (single run): 77.33, +0.57. Pick one | Conflict | Van | Van recommends the two-seed means: eqp_af_lora16_e3 77.43, +0.66 over eqp_af_bn_e3 76.77. Single run af_lora_r16_e3 77.33 (+0.57 vs af_bn_bias_norm_e3 76.756) is within 0.11 |
| P10 | CodaBench build path | CodaBench Phase 1 | 77.9 (336 px) → 79.4 (+ biases/norms, 384 px) → 80.4 (448 px) → 81.2 (+ LoRA) → 82.2 (+ zoom) | To verify | Van | CodaBench IDs: 971289 (336, 0.779), 994764 (384 + biases/norms, 0.794), 1000827 (448, 0.804), 1008585 (+ LoRA, 0.812), 1008590 (crop 0.9 + zoom, 0.822). Final run CS604_RUNS/van_adaptformer_20261021_083823 |
| P12 | Fit vs generalisation: AdaptFormer vs full FT | Train images vs holdout | 95.0 vs 95.5 on train; 1.4 lower on holdout. Runs not identified | Verified | Van | eval_fit on best.pt of af_bn_bias_norm_e3 vs full_proto_336_len3 (336 px, 3 ep). Train (5/class seen) 94.97 vs 95.46; holdout 77.45 vs 78.81 (crop 0.75). Record found |
| P23 | LoRA rank beside AdaptFormer: r8 vs r16 | Van holdout, 336 px, 1 ep, vs AdaptFormer + biases/norms 72.97 | r8 73.54 (+0.58, 4.61%); r16 73.74 (+0.20 over r8, tie, 4.96%). r16 kept: largest rank under 5%, no speed cost |  | Van | af_lora_e1, af_lora_r16_e1 vs af_bn_bias_norm_e1 (2026-10-20) |

## Prediction-time extensions (both routes)

Zoom and kNN train nothing and apply to both models (report Sections 3.6, 5.4, 5.6).

| # | Ablation | Setup | Result (top-1 %) | Status | Checked by | Run record |
| --- | --- | --- | --- | --- | --- | --- |
| X1 | Second view: flip vs centred box vs aimed zoom | Holdout, full FT, 336 px, 3 ep | 78.80 single; +0.17 / +0.63 / +0.79 | To verify | Van | eval_zoom on full_proto_336_len3, crop 0.75: plain 78.798; + flip 78.970; centred box 79.434; cam zoom 79.588. CodaBench 0.819 vs 0.813 (ID 972132) |
| X2 | kNN variants (6 compared) | Holdout; β chosen on the other half of classes | Mean of 3 closest +1.32 (chosen) down to mean of all +0.13 | To verify |  |  |
| X3 | Learned linear re-ranker on top of kNN | Holdout | +0.29 more; rejected | To verify |  |  |
| X4 | Zoom and kNN together | Holdout, LoRA r24 | 78.31 → 80.27 (zoom +0.7, kNN +1.3, both +2.0) | To verify |  |  |
| X5 | kNN gain on CodaBench | CodaBench Phase 1, same settings | LoRA 336 px +1.40; full FT + zoom +0.85; PEFT + zoom +0.88 | To verify |  |  |
| X6 | Phase 1 vs Phase 2 (overfitting check) | Full FT, 448 px | Zoom 83.22 → 82.91; zoom + kNN 84.07 → 83.84 | To verify |  |  |
| X7 | Test centre-crop fraction + flip (FixRes-style), full FT | Holdout, eval only, 224 px (3 / 6 / 12 ep) and 336 px (3 ep) | 224, 3 ep: crop 0.9 72.98 → crop 0.7 + flip 74.85 (+1.87); +1.51 at 12 ep. 336: 78.39 → crop 0.75 + flip 78.97 (+0.58). CodaBench 224: 0.757 → 0.773 |  | Van | eval_tta on full_proto_headlr10x / len6 / len12 / 336_len3 (analysis/eval_tta*.json). CodaBench IDs 963899 / 964799. LP-FT tuning, Stages 5–6 |
| X8 | Test crop re-tuned at 448 px before zoom: 0.9 vs 0.75 | AdaptFormer 448 screen, chosen on even half of holdout, scored on odd half; then CodaBench for both finals | Odd half +0.28. CodaBench Task 2 0.809 → 0.814; Task 1 0.828 → 0.832 |  | Van | af_res_screen_448/analysis/tta_sweep_448. CodaBench IDs 1001880 / 1002322 (T2), 995951 / 1004167 (T1) |
| X9 | Predicting above the training resolution (384 / 448 views of 336-trained models) | Holdout, eval only, full FT and AdaptFormer trained at 336, 3 ep | At 2 passes vs 336 plain + zoom: full FT +0.11, AdaptFormer −0.04 (ties). Extra 384 views as 3rd/4th pass: +0.3 to +0.5. Dropped |  | Van | eval_res_knn on full_proto_336_len3 and af_bn_bias_norm_e3 (analysis/eval_res_knn.json) |
| X10 | kNN probability blend (Van's variant: votes over own training images, mixed with classifier) | Holdout, 336 px, chosen on even half, scored on odd half | Over plain + zoom: full FT +1.03, AdaptFormer +0.95. Superseded by the team's kNN re-rank (X2) |  | Van | eval_res_knn (same runs as X9) |
| X11 | kNN re-rank weight β re-checked for the 448 zoom submissions | Holdout, 4 models (336 and 448), K 5, m 3, chosen on half the classes | Optimum flat at β 15–30; β 20 within 0.07 of each model's best; β 40 0.1–0.35 worse. β 20 kept |  | Van | eval_rerank_beta (<run>/analysis/rerank_beta.json), 2026-10-30 |
| X12 | Flipped copies added to zoom + kNN (4 passes instead of 2) | Holdout (4 models) and CodaBench Phase 1, both finals | Holdout +0.05 to +0.12. CodaBench Task 1 0.8407 → 0.8415, Task 2 0.8308 → 0.8317. Costs 2x the forward passes |  | Van | eval_rerank_beta --flip; cs604-main scripts/van_predict_rerank_flip.py (van-dev bf8f8f4) |

## Gaps to fill or flag in the paper

These were not run, or the report has no numbers for them. Either find the record or state the limit in the paper.

- [ ] F5: find the head-initialisation run (linear head only) in experiment_index.csv and add it to Table 3 or the appendix.
  **Van:** not mine; I have no linear-head-only run. Related evidence (new row F15): the prototype-initialised cosine head trained alone on frozen features goes 43.90 → 60.37 (10 epochs on cached features).
- [ ] F3 LP-FT: record the probe length and whether the probe stage was inside the 144-minute clock. Only tested at 224 px, 3 epochs, one seed.
  **Van:** the probe was a cosine head (started from the prototypes) trained on cached frozen features for 10 epochs (15,630 updates, batch 256, head lr 3e-3, label smoothing 0.1): about 36 s, plus a one-off 412 s feature cache, on the laptop. It was a laptop screen only, never inside a 144-minute clock, and no final used LP-FT. Seed repeat of the LP-FT arm: 71.37 / 71.03 (seed noise 0.33, row F16).
- [ ] F2 layer decay: no-decay run used the decay-tuned LR, and the 0.75 factor was not tuned. Report wording already says this.
  **Van:** confirmed from the record (`full_lpft_b_nolayerdecay`): same 4e-4 top LR, chosen with decay on, not re-tuned.
- [ ] Fixed settings (label smoothing, drop path, weight decay, clipping, batch 256) were never ablated. Report cites the MAE recipe; confirm shared_vit-b16_augreg2_defaults.yaml matches it.
- [ ] F4: confirm the five-point LR sweep was run with decay on only, and that Appendix A says so.
  **Van:** confirmed. `full_lpft_a_lr*`, 1 epoch, layer decay 0.75 on in every arm, LP-FT head: 5e-5 66.24, 1e-4 66.82, 2e-4 67.22, **4e-4 67.47**, 8e-4 67.34. Flat between 4e-4 and 8e-4; 4e-4 kept. Numbers now in row F4.
- [ ] F13 and P12: Van to share run records.
  **Van:** P12 record found and filled in (`eval_fit` on `af_bn_bias_norm_e3` vs `full_proto_336_len3`). F13 has **no controlled run**: the claim compares my 1-epoch LP-FT run (`full_lpft_a_lr4e-4`, 67.47 on the holdout) with Qian's 11-epoch constant-LR run (0.621 on CodaBench). Different yardsticks, and LR, layer decay, batch, head and regularisation all changed too. Suggest rewording F13 as "recipe change" and citing Qian's own cosine-decay step (0.621 → 0.663) as the closest schedule-only evidence.
  **Van (2026-11-11):** F13 controlled run done. Same recipe as `full_proto_headlr10x` with only the schedule changed: holding the peak LR after warm-up instead of decaying to zero gives 63.99 vs 72.98 at 3 epochs (−8.99, one seed). Suggested report wording: "With everything else fixed, holding the peak learning rate after warm-up instead of decaying it to zero cost 9.0 points at 3 epochs (64.0 vs 73.0 holdout top-1, one seed)." Drop the "1 epoch beat 11 epochs" sentence or mark it as uncontrolled.
- [ ] P9: pick the slides' two-seed means or Van's single runs, and use it everywhere.
  **Van:** use the two-seed means: 77.43 (77.44 / 77.41, `eqp_af_lora16_e3_s0/s1`), +0.66 over AdaptFormer alone (76.77). They are the equal-parameter controlled set; my single run (`af_lora_r16_e3`, 77.33, +0.57) is the older one and lands within 0.11 of them.
- [ ] P8: name the three small design changes that came with the CosFace margin.
  **Van:** adapter input norm2(x) instead of the residual stream (+0.24), PCA-initialised down-projection (+0.08), learned per-block adapter scale (+0.15); each measured alone at 1 epoch, margin 0.1 alone +0.60. Combined at 3 epochs (`af_e_combined_e3`): 76.26 vs 75.27.
- [ ] Loss change to push related classes apart: report says it "did not help in our pilots" but gives no numbers. Add them or drop the claim.
  **Van:** my number: neighbour label smoothing (0.1 mass onto the ±5 neighbouring class IDs), full FT at 384 px, 10 img/class, equal compute: 63.92 vs 64.06 at crop 0.9, −0.14, a tie (`nbr_smooth_384_w5` vs `res_screen_384`; row F19). Anastasia's sibling margin and taxonomy smoothing numbers are in the 2026-11-01 report draft.
- [ ] Seeds: CodaBench runs and most holdout runs use one seed. Already in Limitations; check every "tie" and small gain is hedged to match.
- [ ] @Van do you have comparison between Lora R24 and adaptformer + Lora r16?
  **Van:** not in an identical setup. Closest controlled comparison: LoRA r28 on qkv/proj/fc1/fc2 + biases/norms (the LoRA r24 + extras design at the same 4.94% budget), 77.26, vs AdaptFormer + LoRA r16, 77.43: a tie over two seeds (+0.17; row P7). On CodaBench they were never run at the same resolution.
