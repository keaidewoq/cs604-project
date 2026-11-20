# Team run logs

Every run behind the report, in one folder. Two sources:
1. **Colab runs** from the shared Drive folder `CS604_RUNS`: the final systems, the ablation tables and the analysis (sections below).
2. **Van's laptop runs**, listed in [Van's laptop runs](#vans-laptop-runs).

## Colab runs (shared Drive)

Imported 2026-11-19 from the team's Colab runs, which all wrote to the shared Google Drive folder.
These are the runs behind the final systems, the 1,000-class and 10,000-class ablation tables, and
the error and zoom analysis in the report. Van's laptop runs are in the same folder and are described in [Van's laptop runs](#vans-laptop-runs) below.

**Left out:** checkpoints (`*.pt`, about 9.8 GB), validation prediction files, and image sheets over 1 MB.
The one prediction file kept is the final test submission. Training `.log` files had their
progress-bar lines removed. Everything else is byte-for-byte as the run wrote it. Paths inside the files
still point at `/content/drive/MyDrive/CS604_RUNS/<original folder>`. The original folder name is
listed for each entry below.

### What each run folder contains

| File | Meaning |
| --- | --- |
| `config.yaml` | Resolved config the trainer ran (`python -m src.train`) |
| `epoch_log.csv`, `step_log.csv` | Loss / holdout accuracy per epoch and per logged step |
| `run_stats.txt` | Trainable parameters, wall time, GPU hours, peak VRAM |
| `inference_latency*.json` | Batch-1 / batch-64 latency, plain and with the zoom pass |

Ablation folders are written by `scripts/ablation.py` and add a few files:
- `ablation.csv` is one row per arm: accuracy, delta vs arm A in points, backbone %, VRAM and minutes.
- `ablation_summary.json` holds the spec, the shared base config and the results.
- `arms/<arm>.yaml|.json|.log` are each arm's overrides, result and log.
- `runs/<arm>__<method>_<timestamp>/` holds each arm's trainer output. The arm prefix was added on import, matched from the `run_dir=` line in each arm's log.
- `prototypes*.log/json` record the class-mean head initialisation shared by all arms.

The 1k-class pilots train on 1,000 classes and score on a held-out 5 images per class. All-class runs use
all 10,000 classes, with 40 training images and 5 holdout images per class (50,000 holdout images).

### Where to look for each claim

| Report claim | Folder | Key file → value | Drive original |
| --- | --- | --- | --- |
| **Final Task 1 system:** full FT, 448 px, prototype cosine head, LLRD 0.75, 144 min | `final_full-ft_448px_proto-cosine_llrd0.75_task1-final/` | `run_stats.txt`: 100% trainable, 2.41 GPU-h, 17.8 GB peak; `inference_latency_zoom.json`: zoom latency | `full_ft_20261017_122237` |
| **Final Task 2 system:** AdaptFormer w192 + biases/norms + LoRA r16 qkv, 448 px | `final_adaptformer-w192_lora-r16-qkv_448px_bias-and-layernorm_cosface0.1/` | `run_stats.txt`: 2.40 GPU-h, 14.7 GB peak; `inference_latency*.json` | `van_adaptformer_20261021_083823` |
| CodaBench 0.8407 / 0.8308: crop 0.9 + zoom + kNN re-rank (K=5, m=3, β=20) | `final_zoom-knn-rerank/val_full-ft-448/`, `val_adaptformer-lora-448/` | `rerank_stats.json`: settings, predictions changed per variant, 0.06–0.07 ms per photo re-rank cost | `rerank_submission_van` |
| Submitted test predictions | `final_zoom-knn-rerank/test_full-ft-448/` | `test_team_08_zoom_knn_b20.txt` (the `main_submission` named in `rerank_stats.json`) | `rerank_submission_test_van` |
| LoRA r24 recipe ablation, arms A–I (head start × head lr, rank 24 vs 16, extras, DoRA) | `lora_r24_336px_recipe-ablation_1k-class-pilot/` | `ablation.csv`: A 0.8984, B no extras 0.8970, C rank 16 0.8918, D probe head 0.8692, H linear head from prototypes 0.8892, I DoRA 0.8996 | `ablation_ablation_lora_r24` |
| LoRA depth vs resolution, equal time (fewer blocks, 224 → 336 progressive) | `lora_r24_336px_depth-and-resolution_1k-class-pilot/` | `ablation.csv`: A 0.8986; every J/K/L/M arm lower (0.883–0.890) | `ablation_ablation_lora_depth_res` |
| LoRA spread over all blocks vs concentrated in fewer blocks | `lora_r24_336px_spread-vs-fewer-blocks_1k-class-pilot/` | `ablation.csv`: A (all 12 blocks) 0.8986 best; blocks 4–11 r40 0.8890 | `ablation_ablation_lora_depth_control` |
| Parameter selection at the same budget (whole layers, GPS masks, full FT) | `param-selection_336px_same-budget_1k-class-pilot/` | `ablation.csv`: LoRA A 0.8996 ≥ full FT 0.8960 / 0.8956 ≥ GPS k42 0.8944; `angles.csv` scores the selected layers | `ablation_ablation_param_selection` |
| Controlled PEFT comparison, all 10,000 classes, 20 min each, 3 seeds for LoRA and AdaptFormer | `peft_336px_controlled-comparison_all-10000-classes/` | **`ablation_all_arms.csv`**: LoRA 0.7395/0.7399/0.7407, AdaptFormer 0.7400/0.7377/0.7408, Adapter 0.7366, LoRA blocks 4–11 0.7378, BitFit 0.7186, Prefix 0.6820, last-block attention 0.6547 | `ablation_ablation_peft_controlled` |
| LoRA r24 no extras on all classes, 144 min: the model the error analysis uses | `lora_r24_336px_no-extras_all-10000-classes_40fit-5holdout_144min/` | `ablation.csv`: holdout 0.7779 | `ablation_allclass_holdout_lora_no_extras` |
| Top-k, error causes, kNN re-rank by cause | `analysis_error-causes-and-knn-rerank_all-10000-classes/` | `errors/summary.json`: top-1 0.7778, top-5 0.9203, 37% of errors have the answer as second guess; `causes/`, `rerank*/` | `allclass_holdout_analysis` |
| Zoom box search on the same model | `analysis_zoom-boxes_lora-r24-336px/` | `lora_336/summary.json`: top-1 0.7831 → 0.7904 (+0.73), 1,212 fixed / 846 broken | `zoom_boxes` |
| Section 3.4: 79% of images have a short side of 300–400 px | `analysis_dataset-inspection_image-sizes-and-duplicates/` | `image_stats.csv` (10,000-image sample): 79.1%; also class counts and duplicate checks | `dataset_inspection` |

**PEFT method table (Table 3, validation at 336 px, 144 min).** Top-1 is the CodaBench score, recorded in
[`../../experiment_index.csv`](../../experiment_index.csv). VRAM and latency come from each run's files.

| Row | Folder | VRAM / batch-1 latency in the files | Drive original |
| --- | --- | --- | --- |
| Last-block attention 67.93 | `last_1block_attention_336px_proto-linear-head_2.4h/` | 1.43 GiB / 7.76 ms | `last_blocks_20261010_022115` |
| Prefix, length 50: 72.1 | `prefix_336px_len50_all-blocks_linear-proto-head_2.4h/`; length and placement pilot in `prefix-tuning_336px_length-and-placement_pilot/` | 7.97 / 9.64 | `prefix_20261010_000404_id974631`, `qq_prefix_ablation_336_seed42_v1` |
| BitFit 73.4 | `bitfit_336px_all-biases_linear-proto-head_2.4h/`; lr and scope pilot in `bitfit_336px_all-biases_lr-and-scope_pilot/` | 7.01 / 7.40 | `bitfit_20261011_024724_id976355`, `qq_bitfit_ablation_336_subset_seed42_v2` |
| Serial adapter w96 + extras 77.6 | `adapter_w96_336px_extras_proto-cosine_2.4h/`; lr pilot in `adapter_w96_336px_lr-pilot/` | 15.54 / 11.60 | `adapter_20261009_131150`, `pilot_adapter_336_20261009_125653` |
| GPS 77.9 | `gps_top42-weights_336px_extras_proto-cosine_2.4h/` | 20.37 / 7.13 | `selected_20261019_072529` |
| AdaptFormer w192: latency 10.28 | `adaptformer_w192_336px_bias-and-layernorm_cosface0.1_2.4h/` (the same design plus biases/norms; CodaBench 0.802 with zoom) | 15.61 / 10.28 | `van_adaptformer_20261012_060248` |
| LoRA r24 78.4 | `lora_r24_336px_qkv-proj-fc_no-extras_proto-cosine_2.4h/` | 19.30 / 15.37 | `lora_20261020_062136` |
| LoRA r24 + extras 78.5 | `lora_r24_336px_qkv-proj-fc_extras_proto-cosine_2.4h/` | 19.40 / 14.98 | `lora_20261007_051926` |
| AdaLoRA 2.7 pp below LoRA (1k-class pilot, 448 px, 12 min) | `adalora-vs-lora-r24_448px_init-rank24_1k-class-ablation/` | `comparison.json`: LoRA 0.9076; AdaLoRA 0.8792 / 0.8794 / 0.8804 (−2.72 to −2.84) | `qq_adalora_448_c1000_seed42_v1` |

**Validation progressions (Sections 5.2 and 5.3).**

| Report step | Folder | Drive original |
| --- | --- | --- |
| Full FT 66.3% (schedule decaying to zero, linear prototype head, 224 px) | `full_ft_224px_proto-linear-head_backbone-lr1e-5_head-lr1e-3_2.4h/` | `full_ft_20261005_074553_id962394` |
| Full FT 75.7% (Section 3.3 recipe, 224 px) | `full_ft_224px_proto-cosine_llrd0.75_top-lr4e-4_2.4h/` | `full_ft_20261006_031830` |
| Full FT 81.7% (384 px) | `full_ft_384px_proto-cosine_llrd0.75_fused-adamw_compile_2.4h/` (25.34 GiB, 7.9 epochs) | `full_ft_20261016_052544` |
| PEFT 79.4% (+ biases/norms, 384 px) | `adaptformer_w192_384px_bias-and-layernorm_fused-adamw_compile_2.4h/` | `van_adaptformer_20261016_142837` |
| PEFT 80.4% (448 px) | `adaptformer_w192_448px_bias-and-layernorm_fused-adamw_compile_2.4h/` | `van_adaptformer_20261018_073912` |

Two imported runs are not cited in the report: `full_ft_336px_proto-cosine_llrd0.75_fused-adamw_compile_2.4h/`
(Van's 336 px rerun of the recipe, `full_ft_20261015_062349`) and
`full_ft_448px_proto-cosine_llrd0.75_25pct-full-image_2.4h/` (448 px with 25% whole-image crops,
`full_ft_20261026_112113`).

The pilot and ablation folders from Qian's launcher (`prefix-tuning_*_pilot`, `bitfit_*_pilot`, `adalora-*`)
use a different layout from `scripts/ablation.py`. `pilot_results.csv` / `results.csv` hold one row per arm,
`pilot_selection.json` / `comparison.json` record the choice, and `runs/<stage>/<arm>/seed42/` holds each arm's trainer output.
Their data-split lists (`splits/`, `split_manifests/`) were left out, as they are for every other run.

`peft_336px_controlled-comparison_all-10000-classes/ablation.csv` lists only the three arms from the
last Colab session, because each session rewrote the file. `ablation_all_arms.csv` was compiled on import from
`arms/*.json`, which holds every arm's result.

### Claims with a written record but no raw logs

These numbers are written up in [`../../progress_log.md`](../../progress_log.md) or
[`EXPERIMENTS_van.md`](EXPERIMENTS_van.md), but their run folders are not in the shared Drive folder:

| Report claim | Expected folder | Written record |
| --- | --- | --- |
| kNN scoring variants (Table A3) and zoom + kNN on the LoRA holdout, 78.31 → 80.27 | `CS604_RUNS/rerank_bakeoff/` | progress_log, 2026-10-27, "which re-ranker is best?" |
| PEFT table: DoRA r24 + extras 78.3 | `CS604_RUNS/lora_20261012_130611` | progress_log, 2026-10-13 |
| PEFT table: AdaptFormer w192 77.9, 14.53 GiB | Van's `af_final_45` run (his own pipeline, on Colab) | EXPERIMENTS_van.md, "Task 2 AdaptFormer: final run and CodaBench (2026-10-09)" |
| Full FT 80.7% at 336 px | `CS604_RUNS/full_ft_20261006_122039` | progress_log, 2026-10-06 |
| Linear probe on frozen features 62.5% | Anastasia's linear-probe notebook | progress_log, 2026-10-04 |
| Initial full FT 61.3% and 62.1% | SMU cluster jobs 953226 and 959941 | progress_log, Qian, 2026-09-30 to 10-02 |

## Van's laptop runs

Van ran these on his laptop with his own experiment harness, not on Colab, so they were never in the Drive folder.
They were copied into the repo on 2026-11-11 and moved here on 2026-11-19. They back the full fine-tuning
findings (F1–F19) and the PEFT findings (P3–P12, X1–X12). Checkpoints, data splits and prediction files
were left out. Paths inside the files still point to Van's laptop.
[`EXPERIMENTS_van.md`](EXPERIMENTS_van.md) is Van's written record. Each entry names its runs, recipe and
conclusion, using the original short ids, which the last table below maps to folders.

These folders use a different file layout from the Colab runs above:

| File | Meaning |
| --- | --- |
| `run.json` | Final summary: `best_internal_accuracy` (holdout top-1 as a fraction), parameter counts (`backbone_trainable_params`, `backbone_trainable_pct`), GPU, time, git commit |
| `metrics.csv` | Every logged step; rows with `phase = val` are the per-epoch holdout scores |
| `config.json` | Full resolved config |
| `baseline.json` | Step-zero check: accuracy before any update. Every run in a comparison must start from the same number |
| other `.json` / `.csv` | Evaluation-only results (zoom, kNN, fit check, error causes) |

Holdout uses all 10,000 classes, with 40 training images and 5 held out per class (50,000 images, split seed 604). Differences under 0.4 points are ties.
`equal-params_adaptformer-w192_bias-norm_336px_3epoch_seed1` has no `metrics.csv`. Drive filled up during that run, so its score is only in `run.json`.

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
| X1-X12: zoom, kNN, flip | `eval_zoom-plus-knn_add-horizontal-flip`, `eval_zoom-plus-knn_add-horizontal-flip_smoke`, `final_adaptformer-w192_lora-r16-qkv_448px_bias-and-layernorm_cosface0.1` |
| Task 2 CodaBench final, 2026-10-21 | `final_adaptformer-w192_lora-r16-qkv_448px_bias-and-layernorm_cosface0.1` (see the table above) |

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
| `van_adaptformer_20261021_083823` | `final_adaptformer-w192_lora-r16-qkv_448px_bias-and-layernorm_cosface0.1` |
