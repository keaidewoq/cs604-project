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

## Qian's LoRA crop/resolution experiment

On `qq_lora_ablation`, `notebooks/colab_launcher_qq.ipynb` runs the LoRA r24 + extras recipe at 336 and 384, each with the current crops or a 75% crop / 25% whole-image resize-and-pad mixture. It uses the official backbone, a shared 1,000-class fit/holdout split, fitting-only prototypes per resolution, a 10,000-way classifier, and 12 training minutes per arm by default. Evaluation uses crop 0.9 without flip or zoom; selection uses final accuracy after equal training time.

The optional final comparison reuses the better 384 arm and runs matching settings at 448 only if 384 beats the best 336 result by more than 0.3 percentage points. Results, configs, checkpoints, split lists and plots go to Drive. Select an A100 GPU runtime. A–D use 48 training minutes and the optional 448 arm adds 12; setup, prototype extraction and evaluation take additional wall time. This is a subset screen, not a final 144-minute all-data run.

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

Extra knobs: `head_type` / `head_init` / `prototype_file` (Van), `head_lr` (Qian: backbone keeps `lr`, classifier uses `head_lr`), `last_n_blocks`, `evaluate_before_training`.

Experiment recipes: `configs/full_ft_proto.yaml` (Qian’s prototype-initialized full FT), `augreg_in21k.yaml` (Van’s stronger 21k-only prototype backbone). Team numbers live in `report/experiment_index.csv`.

On Colab, `colab_launcher.ipynb` also prints the **resolved** config for your chosen `METHOD` as a table before training. After a run, add a line to `report/experiment_index.csv` and a note in `report/progress_log.md`.

## Add a method or an ablation

### Qian's LoRA versus DoRA screen

On `qq_lora_ablation`, open `notebooks/colab_launcher_qq_lora_dora.ipynb` on
an A100. It uses crop/resolution arm E's 448px recipe, fitting-only prototypes
and a fixed 1,000-class split. Each arm gets 12 training minutes. The preflight
counts the actual ViT parameters and chooses the largest shared rank below 5%:
LoRA r24 is the reference; if needed, LoRA and DoRA at a reduced shared rank
provide the controlled method comparison. Classifier parameters are excluded.
At 448px, microbatch 128 exceeded A100 40GB memory for DoRA. The comparison now
uses microbatch 64 with accumulation 4 for every arm (effective batch 256), and
checks DoRA forward/backward plus AdamW memory before any timed arm starts.
Keep the old 128x2 experiment outputs and use a new folder for the 64x4 comparison.
The primary accuracy comparison is DoRA r23 versus LoRA r24 under the 5% cap;
LoRA r23 is a secondary control.

```bash
python -m src.qq_lora_dora_ablation --preflight-only
python -m src.qq_lora_dora_ablation --memory-check-only
python -m src.qq_lora_dora_ablation --data_root /content/data --output_dir /content/drive/MyDrive/CS604_RUNS/qq_lora_dora_448_c1000_m64_v2 --seeds 42
```

For a second training seed, rerun the same command with `--seeds 42 43` to reuse
completed seed-42 arms. The split and prototype initialization remain at seed 42.
Use a new output folder after code, hardware or recipe changes or an interrupted
arm. Compare final equal-time accuracy, throughput, memory and common completed
epochs in `results.csv` and `comparison.json`. Common-epoch metrics still use
time-based schedules at different LR positions. This subset screen does not
launch full-data training or produce official submissions.

**New method**

1. Add `src/peft/<name>.py` with `apply_<name>(model, cfg)`.
2. Register it in `src/peft/__init__.py` (`build_peft`).
3. Add `configs/<name>.yaml` with `inherits: base.yaml` and `method: <name>`.

`apply_method` already freezes the backbone and leaves the 10k-class head trainable. Your file only needs to attach modules / unfreeze extra tensors.

**New ablation** (same method, different knobs)

Copy an existing method YAML and override only the fields that change (`lora_rank`, `lr`, `epochs`, `model_name`, …). The loader in `src/utils.py` merges `inherits:`.

Adapter and AdaptFormer raise `NotImplementedError` on purpose — see the module docstrings for where the bottleneck should go (after attention/FFN vs parallel to FFN).
