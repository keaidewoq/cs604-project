# CS604 — Fine-Grained ViT Adaptation

Fine-grained multi-class classification (10,000 classes) with a ViT-B/16 foundation model. The course asks for full fine-tuning and a parameter-efficient adaptation method under a 2-hour × 48 GB VRAM training budget.

## Local GPU

The desktop GPU is an RTX 5080 Laptop (16 GB). The default `python` on this machine has a CPU-only PyTorch, which cannot see that GPU. Use the CUDA 13.0 virtualenv outside iCloud:

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\smoke.yaml --synthetic
```

A real run, after `train.tar.gz` is extracted:

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\local.yaml
```

Checkpoints and the loss CSV go to `C:\Users\Van Le\datasets\cs604-outputs\`, not into iCloud. Resume with `--resume C:\Users\Van Le\datasets\cs604-outputs\local-full\last.pt`.

16 GB is smaller than the course examples (24 GB and 48 GB). `configs/local.yaml` uses bf16, gradient checkpointing, and gradient accumulation so ViT-B/16 fits. The report should quote the GPU name and hours from `hardware.json`.

## Colab

`notebooks/colab_train.ipynb` is a launcher. Sign in as `thv.le.2025@mitb.smu.edu.sg`, choose a GPU runtime, and run the cells. Training code stays in `src/`. The notebook clones the repo or copies it from `MyDrive/cs604/group_project`, stages the archives onto `/content`, and calls `scripts/train.py`.

## Layout

```
configs/          experiment settings
src/cs604/        data loading, models, training/eval
scripts/          download and dataset checks
outputs/          checkpoints, logs, CodaBench submissions
notebooks/        exploration
report/           CVPR-format final report
data/             junction to the dataset (not stored in iCloud)
```

## Dataset

Source: [doem1997/cs604-course-data](https://huggingface.co/datasets/doem1997/cs604-course-data)

| Split | Samples | Labels |
| --- | ---: | --- |
| train | 450,000 (45 / class) | `train.txt` |
| val | 50,000 (5 / class) | withheld (Phase 1) |
| test | 100,000 | released in Phase 2 |

The project folder is inside iCloud Drive. The archives are about 45 GB and expand to roughly 500,000 files, so they are stored locally at `C:\Users\Van Le\datasets\cs604-course-data` and linked from `data/`.

```powershell
python scripts/download_dataset.py
python scripts/verify_dataset.py
```

Validation and test labels are withheld. Do not hand-label those images or train on them.

## Submission

One line per image, `image_path class_id`, with `class_id` in `[0, 9999]`. Name the file `val_team_xx.txt` (Phase 1) or `test_team_xx.txt` (Phase 2), zip it, and upload to CodaBench. Write submissions under `outputs/submissions/`.
