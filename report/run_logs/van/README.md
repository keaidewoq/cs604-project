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

Folder names describe the config. `EXPERIMENTS.md` still uses the original short ids
in the first column. `config.json` inside each folder still records the laptop path.

| Claim | Folder |
| --- | --- |
| F13 / Section 3.3: cosine decay to zero vs constant lr | `full-ft_224px_prototype-cosine_llrd0.75_head-lr-10x_cosine-to-zero_3epoch`, `full-ft_224px_prototype-cosine_llrd0.75_constant-lr_3epoch` |
| F1-F3: head lr, layer decay, prototype head vs LP-FT | `full-ft_224px_lpft-head_llrd0.75_head-lr-equals-backbone_3epoch`, `full-ft_224px_lpft-head_llrd0.75_head-lr-10x_3epoch_seed0`, `full-ft_224px_lpft-head_no-layer-decay_top-lr4e-4_3epoch`, `full-ft_224px_prototype-cosine_llrd0.75_head-lr-10x_cosine-to-zero_3epoch` |
| F4: top lr sweep, 1 epoch | `full-ft_224px_lpft-head_llrd0.75_top-lr5e-5_1epoch`, `full-ft_224px_lpft-head_llrd0.75_top-lr1e-4_1epoch`, `full-ft_224px_lpft-head_llrd0.75_top-lr2e-4_1epoch`, `full-ft_224px_lpft-head_llrd0.75_top-lr4e-4_1epoch`, `full-ft_224px_lpft-head_llrd0.75_top-lr8e-4_1epoch` |
| F6: 6 / 12 epochs | `full-ft_224px_prototype-cosine_llrd0.75_6epoch`, `full-ft_224px_prototype-cosine_llrd0.75_12epoch` |
| F7: 336 px, 3 epochs | `full-ft_336px_prototype-cosine_llrd0.75_head-lr-10x_3epoch` |
| F8-F9: resolution at equal compute, and 224 then 384 | `full-ft_336px_equal-compute_10img-per-class`, `full-ft_384px_equal-compute_10img-per-class`, `full-ft_448px_equal-compute_10img-per-class`, `full-ft_512px_equal-compute_10img-per-class`, `full-ft_384px_first-75pct-steps-at-224_equal-compute_10img` |
| F11: prototype accuracy before training | `class-prototypes_augreg2-ft-in1k_224px_no-training`, `class-prototype-bank_224px`, `class-prototypes_augreg-in21k-only_224px_no-training`, `class-prototypes_dinov2-vit-b14_224px_no-training`, `class-prototypes_original-in21k_224px_no-training` |
| F12: wider crop, no colour jitter, 30x head lr | `full-ft_224px_lpft-head_crop-scale-0.35-to-1_3epoch`, `full-ft_224px_lpft-head_no-colour-jitter_3epoch`, `full-ft_224px_lpft-head_head-lr-30x_3epoch`, `full-ft_224px_prototype-cosine_llrd0.75_head-lr-30x_3epoch` |
| F15: head alone on frozen features | `frozen-augreg2-features_cache_for-head-sweep`, `frozen-backbone_linear-head-lr-sweep_augreg2` |
| F16: seed repeat | `full-ft_224px_lpft-head_llrd0.75_head-lr-10x_3epoch_seed0`, `full-ft_224px_lpft-head_llrd0.75_head-lr-10x_3epoch_seed1` |
| F19: neighbour label smoothing | `full-ft_384px_neighbour-label-smoothing-window5_equal-compute_10img` vs `full-ft_384px_equal-compute_10img-per-class` |
| P5-P7: equal-parameter controls, two seeds | `equal-params_adaptformer-w192_lora-r16_336px_3epoch_seed0`, `equal-params_adaptformer-w224_bias-norm_336px_3epoch_seed0`, `equal-params_lora-r28_all-linears_bias-norm_336px_3epoch_seed0`, `equal-params_adaptformer-w192_bias-norm_336px_3epoch_seed0` and the matching `seed1` folders |
| P4: spread vs last 8 blocks | `equal-params_lora-r42_blocks4-11_336px_3epoch_seed0`, `equal-params_adaptformer-w288_blocks4-11_336px_3epoch_seed0` and the matching `seed1` folders |
| P3, P8, P9: AdaptFormer build path | `adaptformer-w192_336px_plain-adapter_3epoch`, `adaptformer-w192_336px_norm-pca-scale-margin_3epoch`, `adaptformer-w192_336px_bias-and-layernorm_3epoch`, `adaptformer-w192_lora-r16-qkv_336px_bias-norm_3epoch` |
| P12: fit vs generalisation | `adaptformer-w192_336px_bias-and-layernorm_3epoch`, `full-ft_336px_prototype-cosine_llrd0.75_head-lr-10x_3epoch` |
| P11: AdaptFormer resolution | `adaptformer-w192_336px_bias-norm_equal-compute_10img-per-class`, `adaptformer-w192_384px_bias-norm_equal-compute_10img-per-class`, `adaptformer-w192_448px_bias-norm_equal-compute_10img-per-class` |
| X1-X12: zoom, kNN, flip | `eval_zoom-plus-knn_add-horizontal-flip`, `eval_zoom-plus-knn_add-horizontal-flip_smoke`, `task2-codabench_adaptformer-w192_lora-r16_448px_2026-10-21` |
| Task 2 CodaBench final, 2026-10-21 | `task2-codabench_adaptformer-w192_lora-r16_448px_2026-10-21` |

Paths inside the files point to Van's laptop; they are kept unchanged as part of the record.
`equal-params_adaptformer-w192_bias-norm_336px_3epoch_seed1` has no `metrics.csv`: Drive filled up during that Colab run, so its score is in `run.json` only.

**Original id to folder**

| Original id | Folder |
| --- | --- |
| `af_a_lr1e-2` | `adaptformer-w192_336px_lr1e-2_plain-adapter_1epoch` |
| `af_a_lr1e-3` | `adaptformer-w192_336px_lr1e-3_plain-adapter_1epoch` |
| `af_a_lr3e-3` | `adaptformer-w192_336px_lr3e-3_plain-adapter_1epoch` |
| `af_a_lr3e-4` | `adaptformer-w192_336px_lr3e-4_plain-adapter_1epoch` |
| `af_adrop01_e3` | `adaptformer-w192_336px_bias-norm_adapter-dropout0.1_3epoch` |
| `af_aug_ra_re_e3` | `adaptformer-w192_336px_bias-norm_randaugment-erasing_3epoch` |
| `af_b_norm` | `adaptformer-w192_336px_adapter-reads-norm_1epoch` |
| `af_b_pca` | `adaptformer-w192_336px_pca-init_1epoch` |
| `af_b_scale` | `adaptformer-w192_336px_learned-scale_1epoch` |
| `af_bn_bias_e1` | `adaptformer-w192_336px_trainable-biases_1epoch` |
| `af_bn_bias_norm_e1` | `adaptformer-w192_336px_bias-and-layernorm_1epoch` |
| `af_bn_bias_norm_e3` | `adaptformer-w192_336px_bias-and-layernorm_3epoch` |
| `af_bn_biaslr3e-3_e1` | `adaptformer-w192_336px_bias-lr3e-3_1epoch` |
| `af_bn_control_e1` | `adaptformer-w192_336px_recipe-no-bias-norm_1epoch` |
| `af_bn_lr3e-3_e1` | `adaptformer-w192_336px_bias-norm_adapter-lr3e-3_1epoch` |
| `af_c_ls0` | `adaptformer-w192_336px_no-label-smoothing_1epoch` |
| `af_c_margin01` | `adaptformer-w192_336px_cosface-margin0.1_1epoch` |
| `af_c_margin02` | `adaptformer-w192_336px_cosface-margin0.2_1epoch` |
| `af_d_w64` | `adaptformer-w64_336px_plain-adapter_1epoch` |
| `af_e_combined_e3` | `adaptformer-w192_336px_norm-pca-scale-margin_3epoch` |
| `af_e_control_e3` | `adaptformer-w192_336px_plain-adapter_3epoch` |
| `af_lora_e1` | `adaptformer-w192_lora-r8-qkv_336px_bias-norm_1epoch` |
| `af_lora_r16_e1` | `adaptformer-w192_lora-r16-qkv_336px_bias-norm_1epoch` |
| `af_lora_r16_e3` | `adaptformer-w192_lora-r16-qkv_336px_bias-norm_3epoch` |
| `af_res_screen_336` | `adaptformer-w192_336px_bias-norm_equal-compute_10img-per-class` |
| `af_res_screen_384` | `adaptformer-w192_384px_bias-norm_equal-compute_10img-per-class` |
| `af_res_screen_448` | `adaptformer-w192_448px_bias-norm_equal-compute_10img-per-class` |
| `af_ssf_e1` | `adaptformer-w192_336px_ssf-instead-of-bias-norm_1epoch` |
| `af_width_graded_e1` | `adaptformer_336px_widths-96-192-288-by-depth_1epoch` |
| `eqp_af224_e3_s0` | `equal-params_adaptformer-w224_bias-norm_336px_3epoch_seed0` |
| `eqp_af224_e3_s1` | `equal-params_adaptformer-w224_bias-norm_336px_3epoch_seed1` |
| `eqp_af288_b4_e3_s0` | `equal-params_adaptformer-w288_blocks4-11_336px_3epoch_seed0` |
| `eqp_af288_b4_e3_s1` | `equal-params_adaptformer-w288_blocks4-11_336px_3epoch_seed1` |
| `eqp_af_bn_e3_s0` | `equal-params_adaptformer-w192_bias-norm_336px_3epoch_seed0` |
| `eqp_af_bn_e3_s1` | `equal-params_adaptformer-w192_bias-norm_336px_3epoch_seed1` |
| `eqp_af_lora16_e3_s0` | `equal-params_adaptformer-w192_lora-r16_336px_3epoch_seed0` |
| `eqp_af_lora16_e3_s1` | `equal-params_adaptformer-w192_lora-r16_336px_3epoch_seed1` |
| `eqp_lora28_e3_s0` | `equal-params_lora-r28_all-linears_bias-norm_336px_3epoch_seed0` |
| `eqp_lora28_e3_s1` | `equal-params_lora-r28_all-linears_bias-norm_336px_3epoch_seed1` |
| `eqp_lora42_b4_e3_s0` | `equal-params_lora-r42_blocks4-11_336px_3epoch_seed0` |
| `eqp_lora42_b4_e3_s1` | `equal-params_lora-r42_blocks4-11_336px_3epoch_seed1` |
| `feature_cache` | `frozen-augreg2-features_cache_for-head-sweep` |
| `full_lpft_a_lr1e-4` | `full-ft_224px_lpft-head_llrd0.75_top-lr1e-4_1epoch` |
| `full_lpft_a_lr2e-4` | `full-ft_224px_lpft-head_llrd0.75_top-lr2e-4_1epoch` |
| `full_lpft_a_lr4e-4` | `full-ft_224px_lpft-head_llrd0.75_top-lr4e-4_1epoch` |
| `full_lpft_a_lr5e-5` | `full-ft_224px_lpft-head_llrd0.75_top-lr5e-5_1epoch` |
| `full_lpft_a_lr8e-4` | `full-ft_224px_lpft-head_llrd0.75_top-lr8e-4_1epoch` |
| `full_lpft_b_control` | `full-ft_224px_lpft-head_llrd0.75_head-lr-equals-backbone_3epoch` |
| `full_lpft_b_headlr10x` | `full-ft_224px_lpft-head_llrd0.75_head-lr-10x_3epoch_seed0` |
| `full_lpft_b_headlr10x_seed1` | `full-ft_224px_lpft-head_llrd0.75_head-lr-10x_3epoch_seed1` |
| `full_lpft_b_nolayerdecay` | `full-ft_224px_lpft-head_no-layer-decay_top-lr4e-4_3epoch` |
| `full_lpft_c_crop035` | `full-ft_224px_lpft-head_crop-scale-0.35-to-1_3epoch` |
| `full_lpft_c_headlr30x` | `full-ft_224px_lpft-head_head-lr-30x_3epoch` |
| `full_lpft_c_nojitter` | `full-ft_224px_lpft-head_no-colour-jitter_3epoch` |
| `full_proto_336_len3` | `full-ft_336px_prototype-cosine_llrd0.75_head-lr-10x_3epoch` |
| `full_proto_constlr` | `full-ft_224px_prototype-cosine_llrd0.75_constant-lr_3epoch` |
| `full_proto_headlr10x` | `full-ft_224px_prototype-cosine_llrd0.75_head-lr-10x_cosine-to-zero_3epoch` |
| `full_proto_headlr30x` | `full-ft_224px_prototype-cosine_llrd0.75_head-lr-30x_3epoch` |
| `full_proto_len12` | `full-ft_224px_prototype-cosine_llrd0.75_12epoch` |
| `full_proto_len6` | `full-ft_224px_prototype-cosine_llrd0.75_6epoch` |
| `head_sweep_augreg2` | `frozen-backbone_linear-head-lr-sweep_augreg2` |
| `nbr_smooth_384_w5` | `full-ft_384px_neighbour-label-smoothing-window5_equal-compute_10img` |
| `prototype-head-screen` | `prototype-cosine-head_lr-screen_224px_frozen-backbone` |
| `prototype-lora-r8-fp32-40` | `lora-r8_224px_frozen-prototype-head_fp32_short-screen` |
| `prototype-lora-r8-fp32-screen` | `lora-r8_224px_frozen-prototype-head_fp32_10img-screen` |
| `prototype-validate-fp32-head` | `class-prototypes_augreg2-ft-in1k_224px_no-training` |
| `prototype_lora_r8_fp32_budget_40` | `lora-r8_224px_frozen-prototype-head_40min-budget` |
| `prototype_validate_augreg_in21k` | `class-prototypes_augreg-in21k-only_224px_no-training` |
| `prototype_validate_dinov2` | `class-prototypes_dinov2-vit-b14_224px_no-training` |
| `prototype_validate_orig_in21k` | `class-prototypes_original-in21k_224px_no-training` |
| `prototypes` | `class-prototype-bank_224px` |
| `prototypes_336` | `class-prototype-bank_336px` |
| `prototypes_384` | `class-prototype-bank_384px` |
| `prototypes_448` | `class-prototype-bank_448px` |
| `prototypes_512` | `class-prototype-bank_512px` |
| `rerank_flip` | `eval_zoom-plus-knn_add-horizontal-flip` |
| `rerank_flip_test` | `eval_zoom-plus-knn_add-horizontal-flip_smoke` |
| `res_screen_224to384` | `full-ft_384px_first-75pct-steps-at-224_equal-compute_10img` |
| `res_screen_336` | `full-ft_336px_equal-compute_10img-per-class` |
| `res_screen_384` | `full-ft_384px_equal-compute_10img-per-class` |
| `res_screen_448` | `full-ft_448px_equal-compute_10img-per-class` |
| `res_screen_512` | `full-ft_512px_equal-compute_10img-per-class` |
| `ssf_lr1e-2_e1` | `ssf_336px_lr1e-2_cosface0.1_1epoch` |
| `ssf_lr1e-3_e1` | `ssf_336px_lr1e-3_cosface0.1_1epoch` |
| `ssf_lr3e-3_e1` | `ssf_336px_lr3e-3_cosface0.1_1epoch` |
| `van_adaptformer_20261021_083823` | `task2-codabench_adaptformer-w192_lora-r16_448px_2026-10-21` |
