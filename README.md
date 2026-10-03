# CS604 — Fine-Grained ViT Adaptation

Fine-grained multi-class classification (10,000 classes) with a ViT-B-sized vision transformer (currently ViT-B/16). The course asks for full fine-tuning and a parameter-efficient adaptation method under a 2-hour × 48 GB VRAM training budget.

**Course rules confirmed 2026-10-03** (details in `AGENTS.md`): the classifier head is always trained and is not counted in the 5% limit; any comparable-size ViT may be the backbone; no checkpoints trained on iNaturalist 2021 or TreeOfLife-10M. Current status: `report/HANDOFF.md`.

## Local GPU

This repository is `C:\Users\Van Le\cs604-project\cs604-group-project`, outside iCloud and OneDrive. The desktop GPU is an RTX 5080 Laptop (16 GB). The default `python` on this machine has a CPU-only PyTorch, which cannot see that GPU. Use the CUDA 13.0 virtualenv at `C:\Users\Van Le\.venvs\cs604`:

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\smoke.yaml --synthetic
```

A real run reads the sibling dataset and writes checkpoints beside this repo:

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\local.yaml
```

Checkpoints and the loss CSV go to `C:\Users\Van Le\cs604-project\cs604-outputs\`. Resume with `--resume C:\Users\Van Le\cs604-project\cs604-outputs\local-full\last.pt`.

16 GB is smaller than the course examples (24 GB and 48 GB). `configs/local.yaml` uses bf16, gradient checkpointing, and gradient accumulation so ViT-B/16 fits. The report should quote the GPU name and hours from `hardware.json`.

## Colab

`notebooks/colab_train.ipynb` is a launcher. Sign in as `thv.le.2025@mitb.smu.edu.sg`, choose a GPU runtime, and run the cells. Training code stays in `src/`. The notebook clones the repo or copies it from `MyDrive/cs604/group_project`, stages the archives onto `/content`, and calls `scripts/train.py`.

## Layout

```
C:\Users\Van Le\cs604-project\
  cs604-group-project\   this repository
  cs604-course-data\     images and archives
  cs604-outputs\         checkpoints and logs
```

```
configs/          experiment settings
src/cs604/        data loading, models, training/eval
scripts/          download and dataset checks
outputs/          CodaBench submissions
notebooks/        exploration
report/           CVPR-format final report
```

`data/` links to `C:\Users\Van Le\cs604-project\cs604-course-data`.

## Dataset

Source: [doem1997/cs604-course-data](https://huggingface.co/datasets/doem1997/cs604-course-data)

| Split | Samples | Labels |
| --- | ---: | --- |
| train | 450,000 (45 / class) | `train.txt` |
| val | 50,000 (5 / class) | withheld (Phase 1) |
| test | 100,000 | released in Phase 2 |

The archives are about 45 GB and expand to roughly 500,000 files. They live at `C:\Users\Van Le\cs604-project\cs604-course-data`.

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\download_dataset.py
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\verify_dataset.py
```

Validation and test labels are withheld. Do not hand-label those images or train on them.

## Submission

One line per image, `image_path class_id`, with `class_id` in `[0, 9999]`. Name the file `val_team_xx.txt` (Phase 1) or `test_team_xx.txt` (Phase 2), zip it, and upload to CodaBench. Write submissions into the run folder, `cs604-outputs/<EXPERIMENT>/submissions/` (see `WORKFLOW.md`).

## Experiment workflow

**Step-by-step runbook:** [`WORKFLOW.md`](WORKFLOW.md) covers the virtualenv, local screens,
Colab runs, syncing from Drive, prediction, and recording CodaBench results.

For the current prototype-validation implementation, active worktree, and exact
next-run commands, read [`report/HANDOFF.md`](report/HANDOFF.md). This is the shared
handoff for Codex, Claude Code, and Cursor; use one agent and one GPU job at a time.

Use one YAML file per experiment under `configs/experiments/`. Every run saves
its resolved config, Git commit, hardware, parameter counts, timing, metrics,
and resumable checkpoints.

```powershell
# Compare this exact 100-step job with the Colab benchmark.
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\experiments\benchmark.yaml

# Required full fine-tune and a cheap frozen-backbone baseline.
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\experiments\full_baseline.yaml
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\experiments\head_baseline.yaml
```

The internal split always reserves the same five images per class. Screening
configs use fewer training images per class, never an ordinary random subset.

For the under-5% prototype + LoRA design, first build class prototypes, then
train from that initialization (the head is trained too; the 5% limit counts
backbone parameters only):

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\build_prototypes.py --config configs\experiments\prototype_init.yaml --output C:\Users\Van Le\cs604-project\cs604-outputs\prototypes\class_prototypes.pt
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\experiments\prototype_head.yaml
```

Generate an audited CodaBench file:

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\predict.py --checkpoint C:\path\to\best.pt --data-root C:\Users\Van Le\cs604-project\cs604-course-data --split val --output outputs\submissions\val_team_xx.txt
```

In Colab, set `COMMIT`, `CONFIG`, and `RUN_NAME` in
`notebooks/colab_train.ipynb`. The notebook stages images on `/content` and
keeps checkpoints on Drive. Never train directly from the Drive image tree.

Summarize local and downloaded Colab artifacts with:

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\summarize_experiments.py C:\Users\Van Le\cs604-project\cs604-outputs
```
