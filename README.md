# CS604 course project

10,000-class image classification by fine-tuning ViT-B/16. One trainer (`python -m src.train`) plus YAML configs. Adapter and AdaptFormer are still stubs.

The `van` branch's extra experiment lab was **not** git-merged (it is a second framework). What we kept from it: **prototype / cosine head**, **last-N-block tuning**, and a **shared run log** under `report/`.

Default backbone: `vit_base_patch16_224.augreg2_in21k_ft_in1k` (timm). Change `model_name` in `configs/base.yaml` to try the ImageNet-21k-only variant, e.g. `vit_base_patch16_224.augreg_in21k`.

## Structure

```text
configs/    base.yaml + one file per method / ablation
src/        data, model, heads, losses, train, predict, utils
src/peft/   lora, bitfit, last_blocks; adapter / adaptformer stubs
scripts/    download_data.py, build_prototypes.py, benchmark_latency.py, plot_curves.py
notebooks/  colab_launcher.ipynb
report/     progress_log.md + experiment_index.csv (team log)
outputs/    run dirs (gitignored)
figures/    plots
```

## Run on Colab

1. Open `notebooks/colab_launcher.ipynb` on Colab (GPU runtime, 48GB VRAM if you have it).
2. The notebook clones this repo, installs requirements, mounts Drive, downloads `doem1997/cs604-course-data` to `/content/data`, trains with `output_dir` on Drive, then writes `val_team_xx.txt` / `test_team_xx.txt`.

From the repo root, the same commands are:

```bash
python scripts/download_data.py --dest /content/data
python scripts/build_prototypes.py --config configs/prototype_probe.yaml --output /content/drive/MyDrive/CS604_RUNS/prototypes.pt
python -m src.train --config configs/prototype_lora_r8.yaml --data_root /content/data --output_dir /content/drive/MyDrive/CS604_RUNS
python -m src.predict --config configs/lora_r8.yaml --ckpt /path/to/best.pt --split val --team xx
python scripts/plot_curves.py --run_dir outputs/lora_YYYYMMDD_HHMMSS
python scripts/benchmark_latency.py --config configs/lora_r8.yaml
```

Each run folder stores the resolved `config.yaml`, `step_log.csv`, `epoch_log.csv`, `last.pt`, `best.pt` (when val labels exist), and wall-clock / GPU-hour / peak-VRAM stats.

Submission lines are `image_path class_id`, matching the course label files.

## Parameter table

Shared defaults live in `configs/base.yaml`. A method file only lists what it **changes**. Blank = inherit from base.

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

Extra knobs (from Van): `head_type` (`linear`/`cosine`), `head_init` (`random`/`prototype`), `prototype_file`, `last_n_blocks`, `evaluate_before_training`.

On Colab, `colab_launcher.ipynb` also prints the **resolved** config for your chosen `METHOD` as a table before training. After a run, add a line to `report/experiment_index.csv` and a note in `report/progress_log.md`.

## Add a method or an ablation

**New method**

1. Add `src/peft/<name>.py` with `apply_<name>(model, cfg)`.
2. Register it in `src/peft/__init__.py` (`build_peft`).
3. Add `configs/<name>.yaml` with `inherits: base.yaml` and `method: <name>`.

`apply_method` already freezes the backbone and leaves the 10k-class head trainable. Your file only needs to attach modules / unfreeze extra tensors.

**New ablation** (same method, different knobs)

Copy an existing method YAML and override only the fields that change (`lora_rank`, `lr`, `epochs`, `model_name`, …). The loader in `src/utils.py` merges `inherits:`.

Adapter and AdaptFormer raise `NotImplementedError` on purpose — see the module docstrings for where the bottleneck should go (after attention/FFN vs parallel to FFN).
