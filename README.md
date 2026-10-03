# CS604 course project

10,000-class image classification by fine-tuning ViT-B/16. The skeleton compares full fine-tuning against parameter-efficient methods (LoRA, BitFit; Adapter and AdaptFormer are stubs).

Default backbone: `vit_base_patch16_224.augreg2_in21k_ft_in1k` (timm). Change `model_name` in `configs/base.yaml` to try the ImageNet-21k-only variant, e.g. `vit_base_patch16_224.augreg_in21k`.

## Structure

```text
configs/    base.yaml + one file per method / ablation
src/        data, model, losses, train, predict, utils
src/peft/   lora, bitfit (implemented); adapter, adaptformer (stubs)
scripts/    download_data.py, benchmark_latency.py, plot_curves.py
notebooks/  colab_launcher.ipynb
outputs/    run dirs (gitignored)
figures/    plots
```

## Run on Colab

1. Open `notebooks/colab_launcher.ipynb` on Colab (GPU runtime, 48GB VRAM if you have it).
2. The notebook clones this repo, installs requirements, mounts Drive, downloads `doem1997/cs604-course-data` to `/content/data`, trains with `output_dir` on Drive, then writes `val_team_xx.txt` / `test_team_xx.txt`.

From the repo root, the same commands are:

```bash
python scripts/download_data.py --dest /content/data
python -m src.train --config configs/lora_r8.yaml --data_root /content/data --output_dir /content/drive/MyDrive/cs604-runs
python -m src.predict --config configs/lora_r8.yaml --ckpt /path/to/best.pt --split val --team xx
python scripts/plot_curves.py --run_dir outputs/lora_YYYYMMDD_HHMMSS
python scripts/benchmark_latency.py --config configs/lora_r8.yaml
```

Each run folder stores the resolved `config.yaml`, `step_log.csv`, `epoch_log.csv`, `last.pt`, `best.pt` (when val labels exist), and wall-clock / GPU-hour / peak-VRAM stats.

Submission lines are `image_path class_id`, matching the course label files.

## Add a method or an ablation

**New method**

1. Add `src/peft/<name>.py` with `apply_<name>(model, cfg)`.
2. Register it in `src/peft/__init__.py` (`build_peft`).
3. Add `configs/<name>.yaml` with `inherits: base.yaml` and `method: <name>`.

`apply_method` already freezes the backbone and leaves the 10k-class head trainable. Your file only needs to attach modules / unfreeze extra tensors.

**New ablation** (same method, different knobs)

Copy an existing method YAML and override only the fields that change (`lora_rank`, `lr`, `epochs`, `model_name`, …). The loader in `src/utils.py` merges `inherits:`.

Adapter and AdaptFormer raise `NotImplementedError` on purpose — see the module docstrings for where the bottleneck should go (after attention/FFN vs parallel to FFN).
