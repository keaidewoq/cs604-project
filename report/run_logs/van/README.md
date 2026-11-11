# Van's run logs behind the tracker and report numbers

Copied 2026-11-11 from Van's laptop (`cs604-outputs/`), so teammates can check every number Van
added to the ablation tracker and the final report. One folder per run. Checkpoints, data
splits and CodaBench prediction files are left out; everything else small is kept as written by
the run.

**What to read in each folder**

- `run.json`: final summary: `best_internal_accuracy` (holdout top-1 as a fraction), parameter
  counts (`backbone_trainable_params`, `backbone_trainable_pct`), GPU, time, git commit.
- `metrics.csv`: every logged step; rows with `phase = val` are the per-epoch holdout scores.
- `config.json`: the full resolved config the run used.
- `baseline.json`: the step-zero check (accuracy before any update; every run in a comparison
  must start from the same number).
- Other `.json` / `.csv` files: evaluation-only results (zoom, kNN, fit check, error causes).

`EXPERIMENTS.md` is the written record: each entry names its runs, recipe and conclusion.
Holdout = all 10,000 classes, 40 training images per class, 5 held out (50,000 images, split
seed 604). Differences under 0.4 points are ties.

**Where to look for the main claims**

| Claim (tracker row or report section) | Runs |
| --- | --- |
| F13 / Section 3.3, Table 3: decay to zero vs constant LR | `full_proto_headlr10x`, `full_proto_constlr` |
| F1-F3 / Table 3: head LR, layer decay, prototype head vs LP-FT | `full_lpft_b_control`, `full_lpft_b_headlr10x`, `full_lpft_b_nolayerdecay`, `full_proto_headlr10x` |
| F4: top LR sweep | `full_lpft_a_lr5e-5`, `_lr1e-4`, `_lr2e-4`, `_lr4e-4`, `_lr8e-4` |
| F6: 6 / 12 epochs | `full_proto_len6`, `full_proto_len12` |
| F7: 336 px | `full_proto_336_len3` |
| F8-F9 / Section 5.2: resolution at equal compute, progressive | `res_screen_336` / `384` / `448` / `512`, `res_screen_224to384` |
| F11, F17 / Introduction (43.9%): prototype accuracy | `prototype-validate-fp32-head`, `prototypes_*`, `prototype_validate_*` |
| F12: crop, jitter, 30x head LR | `full_lpft_c_crop035`, `full_lpft_c_nojitter`, `full_lpft_c_headlr30x`, `full_proto_headlr30x` |
| F15: head alone on frozen features | `feature_cache`, `head_sweep_augreg2` |
| F16: seed repeat | `full_lpft_b_headlr10x`, `full_lpft_b_headlr10x_seed1` |
| F19 / Section 6: neighbour label smoothing | `nbr_smooth_384_w5` vs `res_screen_384` |
| P5-P7 / Section 5.3: equal-parameter controls (two seeds) | `eqp_af_lora16_e3_s0/s1`, `eqp_af224_e3_s0/s1`, `eqp_lora28_e3_s0/s1`, `eqp_af_bn_e3_s0/s1` |
| P4a-P4b / Section 5.3: spread vs concentrated | `eqp_lora42_b4_e3_s0/s1`, `eqp_af288_b4_e3_s0/s1` (references above) |
| P3, P8, P9: AdaptFormer build path | `af_e_control_e3`, `af_e_combined_e3`, `af_bn_bias_norm_e3`, `af_lora_r16_e3`, `af_*_e1` screens |
| P12 / Section 5.4: fit vs generalisation | `af_bn_bias_norm_e3`, `full_proto_336_len3` (`analysis/fit_check_crop0.75.json` in each) |
| P11: AdaptFormer resolution | `af_res_screen_336` / `384` / `448` |
| P15-P23: adapter LR, width, loss, regularisation, SSF, LoRA rank | `af_a_*`, `af_d_w64`, `af_c_*`, `af_aug_ra_re_e3`, `af_adrop01_e3`, `af_width_graded_e1`, `af_bn_*lr3e-3_e1`, `af_ssf_e1`, `ssf_lr*_e1`, `af_lora_e1`, `af_lora_r16_e1` |
| X1-X12 / Section 5.6: zoom, kNN, flip | `rerank_flip`, `rerank_flip_test`, `van_adaptformer_20261021_083823` |
| F18: frozen-head LoRA | `prototype-lora-r8-fp32-*`, `prototype_lora_r8_fp32_budget_40`, `prototype-head-screen` |

Paths inside the files point to Van's laptop; they are kept unchanged as part of the record.
`eqp_af_bn_e3_s1` has no `metrics.csv`: Drive filled up during that Colab run, so its score is
in `run.json` only (see `EXPERIMENTS.md`, equal-parameter controls).
