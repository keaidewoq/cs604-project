# CS604 — Fine-Grained ViT Adaptation

Fine-grained multi-class classification (10,000 classes) with a ViT-B-sized vision transformer (currently ViT-B/16). The course asks for full fine-tuning and a parameter-efficient adaptation method under a 2-hour × 48 GB VRAM training budget.

**Course rules confirmed 2026-10-03** (details in `AGENTS.md`): the classifier head is always trained and is not counted in the 5% limit; any comparable-size ViT may be the backbone; no checkpoints trained on iNaturalist 2021 or TreeOfLife-10M. Results so far: `report/EXPERIMENTS.md`.

## Setup

Keep the repo, the dataset, and the run outputs as sibling folders, outside iCloud and OneDrive:

```
cs604-project/
  <this repo>/
  cs604-course-data/     images and archives (~85 GB once extracted)
  cs604-outputs/         checkpoints and logs
```

Configs write these paths as `${CS604_DATA}` and `${CS604_OUTPUTS}`. Unset, they resolve to the two sibling folders above. To keep them somewhere else, set the environment variables, for example in PowerShell:

```powershell
$env:CS604_DATA = "D:\cs604-course-data"
$env:CS604_OUTPUTS = "D:\cs604-outputs"
```

Create a virtualenv with a CUDA build of PyTorch (a CPU-only PyTorch cannot see the GPU), install `requirements.txt`, and check the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The commands below assume that virtualenv is active and the current folder is the repo root.

## Local GPU

Smoke test, no dataset required:

```powershell
python scripts\train.py --config configs\smoke.yaml --synthetic
```

A real run reads the dataset and writes checkpoints and the loss CSV to `cs604-outputs/local-full/`:

```powershell
python scripts\train.py --config configs\local.yaml
```

Resume with `--resume ..\cs604-outputs\local-full\last.pt`.

`configs/local.yaml` targets a 16 GB GPU, smaller than the course examples (24 GB and 48 GB): it uses bf16, gradient checkpointing, and gradient accumulation so ViT-B/16 fits. The report should quote the GPU name and hours from `hardware.json`.

## Colab

`notebooks/colab_train.ipynb` is a launcher. Choose a GPU runtime and run the cells. Training code stays in `src/`. The notebook clones this repo (set `REPO` and `BRANCH` in the checkout cell), stages the archives onto `/content`, and calls `scripts/train.py`.

## Layout

```
configs/          experiment settings
src/cs604/        data loading, models, training/eval
scripts/          download and dataset checks
outputs/          small run records per experiment (see outputs/README.md)
notebooks/        Colab launcher
report/           CVPR-format final report
```

`data/` (gitignored) may link to the dataset folder.

## Dataset

Source: [doem1997/cs604-course-data](https://huggingface.co/datasets/doem1997/cs604-course-data)

| Split | Samples | Labels |
| --- | ---: | --- |
| train | 450,000 (45 / class) | `train.txt` |
| val | 50,000 (5 / class) | withheld (Phase 1) |
| test | 100,000 | released in Phase 2 |

The archives expand to roughly 500,000 files. Download and check them into `cs604-course-data` (or `$CS604_DATA`):

```powershell
python scripts\download_dataset.py
python scripts\verify_dataset.py
```

Validation and test labels are withheld. Do not hand-label those images or train on them.

## Submission

One line per image, `image_path class_id`, with `class_id` in `[0, 9999]`. Name the file `val_team_xx.txt` (Phase 1) or `test_team_xx.txt` (Phase 2), zip it, and upload to CodaBench. Write submissions into the run folder, `cs604-outputs/<EXPERIMENT>/submissions/` (see `WORKFLOW.md`).

## Experiment workflow

**Step-by-step runbook:** [`WORKFLOW.md`](WORKFLOW.md) covers the virtualenv, local screens,
Colab runs, syncing from Drive, prediction, and recording CodaBench results.

Results and decisions are recorded in [`report/EXPERIMENTS.md`](report/EXPERIMENTS.md)
and [`report/progress_log.md`](report/progress_log.md), with one row per run in
`report/experiment_index.csv`. Use one agent and one GPU job at a time.

Use one YAML file per experiment under `configs/experiments/`. Every run saves
its resolved config, Git commit, hardware, parameter counts, timing, metrics,
and resumable checkpoints.

```powershell
# Compare this exact 100-step job with the Colab benchmark.
python scripts\train.py --config configs\experiments\benchmark.yaml

# Required full fine-tune and a cheap frozen-backbone baseline.
python scripts\train.py --config configs\experiments\full_baseline.yaml
python scripts\train.py --config configs\experiments\head_baseline.yaml
```

The internal split always reserves the same five images per class. Screening
configs use fewer training images per class, never an ordinary random subset.

For the under-5% prototype + LoRA design, first build class prototypes, then
train from that initialization (the head is trained too; the 5% limit counts
backbone parameters only):

```powershell
python scripts\build_prototypes.py --config configs\experiments\prototype_init.yaml --output ..\cs604-outputs\prototypes\class_prototypes.pt
python scripts\train.py --config configs\experiments\prototype_head.yaml
```

Generate an audited CodaBench file:

```powershell
python scripts\predict.py --checkpoint C:\path\to\best.pt --data-root ..\cs604-course-data --split val --output ..\cs604-outputs\<EXPERIMENT>\submissions\val_team_xx.txt
```

In Colab, set `COMMIT`, `CONFIG`, and `RUN_NAME` in
`notebooks/colab_train.ipynb`. The notebook stages images on `/content` and
keeps checkpoints on Drive. Never train directly from the Drive image tree.

Summarize local and downloaded Colab artifacts with:

```powershell
python scripts\summarize_experiments.py ..\cs604-outputs
```

## What is in git and what is on Drive

Git holds code, configs, the report records, and each run's small records in
`outputs/<EXPERIMENT>/` (`run.json`, `config.json`, `metrics.csv`, ...), copied
there by `scripts/publish_run.py`. Checkpoints, split lists, and prediction
files go to the team Google Drive folder `CS604_RUNS/<EXPERIMENT>/`. The full
list is in [`outputs/README.md`](outputs/README.md).
