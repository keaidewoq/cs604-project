# CS604 course project

10,000-class image classification by fine-tuning ViT-B/16. One trainer (`python -m src.train`) plus YAML configs. The final report is `report/CS604 Final Report (Team 08) - ICLR.pdf`, built from `report/latex/iclr_report.tex`.

Methods on `main`: full fine-tuning, linear probe, LoRA, BitFit, last-block tuning, serial Adapter, AdaptFormer, prefix tuning, and GPS-style parameter selection (`method: selected`). Visual prompts (VPT) and DoRA are options on the LoRA configs. Van's separate AdaptFormer trainer stays on `van-448-finals`; the AdaptFormer in the report is `src/peft/adaptformer.py`.

Default backbone: `vit_base_patch16_224.augreg2_in21k_ft_in1k` (timm). Change `model_name` in `configs/shared_vit-b16_augreg2_defaults.yaml` to try the ImageNet-21k-only variant, e.g. `vit_base_patch16_224.augreg_in21k`.

## Structure

```text
configs/    shared_vit-b16_augreg2_defaults.yaml + one YAML per experiment
src/        data, model, heads, losses, train, predict, utils
src/peft/   lora, bitfit, last_blocks, adapter, adaptformer, prefix, selected
scripts/    data download, prototypes, sweeps, zoom / kNN prediction, report figures
notebooks/  colab_<experiment>.ipynb, one launcher per config
report/     final report, progress_log.md, experiment_index.csv, run_logs/team/
outputs/    run dirs (gitignored)
figures/    plots
```

A config is named `{method}_{resolution}_{settings}`. Drive run folders (`CS604_RUNS/...`) keep their original names. Notebooks that came from another branch still check that branch out when they run.

## Run on Colab

1. Open the notebook named after the experiment, or `notebooks/colab_qq_train-one-config.ipynb` to train one config (GPU runtime, 48GB VRAM if you have it). `notebooks/colab_van_train-one-config.ipynb` is the launcher for Van's 448 px AdaptFormer run.
2. The notebook clones this repo, installs requirements, mounts Drive, downloads `doem1997/cs604-course-data` to `/content/data`, trains with `output_dir` on Drive, then writes `val_team_xx.txt` / `test_team_xx.txt`.

From the repo root, the same commands are:

```bash
python scripts/download_data.py --dest /content/data
python scripts/build_prototypes.py --config configs/class-prototypes_augreg2_224px_cosine-head.yaml --output /content/drive/MyDrive/CS604_RUNS/prototypes.pt
python -m src.train --config configs/lora_r8_224px_proto-cosine-head.yaml --data_root /content/data --output_dir /content/drive/MyDrive/CS604_RUNS
python -m src.predict --config configs/lora_r8_224px_lr1e-3_5epoch.yaml --ckpt /path/to/best.pt --split val --team xx
python scripts/plot_curves.py --run_dir outputs/lora_YYYYMMDD_HHMMSS
python scripts/benchmark_latency.py --config configs/lora_r8_224px_lr1e-3_5epoch.yaml
```

Each run folder stores the resolved `config.yaml`, `step_log.csv`, `epoch_log.csv`, `last.pt`, `best.pt` (when val labels exist), and wall-clock / GPU-hour / peak-VRAM stats.

Submission lines are `image_path class_id`, matching the course label files.

## Parameter table

Shared defaults live in `configs/shared_vit-b16_augreg2_defaults.yaml`. A method file only lists what it **changes**. Blank = inherit from base.

| Parameter | Meaning | base | full_ft | linear_probe | lora_r8 | bitfit | adapter | adaptformer |
|---|---|---|---|---|---|---|---|---|
| `method` | What to train | `full_ft` | `full_ft` | `linear_probe` | `lora` | `bitfit` | `adapter` | `adaptformer` |
| `model_name` | timm backbone | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | | | | | | |
| `pretrained` | Load ImageNet weights | `true` | | | | | | |
| `num_classes` | Head size | `10000` | | | | | | |
| `img_size` | Train/eval resolution | `224` | | | | | | |
| `data_root` | Where images + `*.txt` live | `/content/data` | | | | | | |
| `batch_size` | Images per step | `64` | `32` | | | | | |
| `grad_accum_steps` | Steps before optimizer update (effective batch = batch × this) | `1` | `2` | | | | | |
| `epochs` | Passes over train | `3` | `2` | `5` | `5` | `5` | `5` | `5` |
| `lr` | AdamW learning rate | `1e-4` | `1e-4` | `1e-3` | `1e-3` | `1e-3` | `1e-3` | `1e-3` |
| `weight_decay` | AdamW weight decay | `0.05` | | `0.0` | | | | |
| `warmup` | Warmup length in epochs | `1.0` | | | | | | |
| `bf16` | Mixed precision on GPU | `true` | | | | | | |
| `num_workers` | DataLoader workers | `2` | | | | | | |
| `seed` | RNG seed | `42` | | | | | | |
| `output_dir` | Run folder parent | `outputs` | | | | | | |
| `label_smoothing` | CE label smoothing | `0.0` | | | | | | |
| `lora_rank` | LoRA rank | — | | | `8` | | | |
| `lora_alpha` | LoRA scale (`alpha/rank`) | — | | | `16` | | | |
| `lora_dropout` | LoRA dropout | — | | | `0.1` | | | |
| `adapter_dim` | Bottleneck width | — | | | | | `64` | `64` |
| `adaptformer_scale` | Parallel FFN scale `s` | — | | | | | | `0.1` |

Extra knobs: `head_type` / `head_init` / `prototype_file` (Van), `head_lr` (Qian: backbone keeps `lr`, classifier uses `head_lr`), `last_n_blocks`, `evaluate_before_training`, `first_trainable_block` (LoRA and `unfreeze_extra` only from that block up; backward stops there), `low_res_size` / `low_res_fraction` (progressive resizing: train at the low size first, finish at `img_size`). Screened in `configs/lora_r24_336px_depth-and-resolution_1k-class-pilot.yaml`. Training pretrained weights directly (`src/peft/selected.py`, alone with `method: selected` or next to LoRA): `unfreeze_modules` / `unfreeze_modules_file` (whole layers at full rank), `sparse_mask_file` (GPS-style masked entries), `unfreeze_lr` (their own lr). `scripts/select_params.py` picks them by fine-tuned angle or gradient; compared in `configs/param-selection_336px_same-budget_1k-class-pilot.yaml`.

Experiment recipes: `configs/full_ft_224px_proto-linear-head_backbone-lr1e-5_head-lr1e-3_2.4h.yaml` (Qian’s prototype-initialized full FT), `class-prototypes_augreg-in21k_224px_cosine-head.yaml` (Van’s stronger 21k-only prototype backbone). Team numbers live in `report/experiment_index.csv`.

On Colab, `colab_qq_train-one-config.ipynb` prints the **resolved** config for the chosen `METHOD` as a table before training. After a run, add a line to `report/experiment_index.csv` and a note in `report/progress_log.md`. Extracted logs for the runs behind the report are indexed in `report/run_logs/team/README.md`.

## Add a method or an ablation

**New method**

1. Add `src/peft/<name>.py` with `apply_<name>(model, cfg)`.
2. Register it in `src/peft/__init__.py` (`build_peft`).
3. Add `configs/<name>.yaml` with `inherits: shared_vit-b16_augreg2_defaults.yaml` and `method: <name>`.

`apply_method` already freezes the backbone and leaves the 10k-class head trainable. Your file only needs to attach modules / unfreeze extra tensors.

**New ablation** (same method, different knobs)

Copy an existing method YAML and override only the fields that change (`lora_rank`, `lr`, `epochs`, `model_name`, …). The loader in `src/utils.py` merges `inherits:`. Name the file after the experiment.

Adapter (`src/peft/adapter.py`) is a serial bottleneck after attention and after the MLP. AdaptFormer (`src/peft/adaptformer.py`) is a parallel bottleneck on the MLP, scaled by `adaptformer_scale`. Both are registered in `src/peft/__init__.py`.
