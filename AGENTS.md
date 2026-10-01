# CS604 agent instructions

Fine-grained 10,000-class classification with ViT-B/16. The course requires a full fine-tune and a parameter-efficient method, under a 2-hour × 48 GB training budget.

## Paths

This git repo is `C:\Users\Van Le\cs604-project\cs604-group-project`.

| What | Where |
| --- | --- |
| Python (CUDA) | `C:\Users\Van Le\.venvs\cs604\Scripts\python.exe` |
| Dataset | `C:\Users\Van Le\cs604-project\cs604-course-data` |
| Checkpoints and logs | `C:\Users\Van Le\cs604-project\cs604-outputs` |
| CodaBench files | `outputs/submissions/` inside this repo |

The default `python` on this machine is CPU-only and cannot see the RTX 5080. Always call the virtualenv interpreter above. Run commands from the repo root.

## Hard rules

- Validation and test labels are withheld. Do not hand-label those images, train on them, or infer labels from filenames.
- Keep the internal split fixed: the same five images per class. Screening configs use fewer training images per class (5, then 10, then 20). Never replace that with an ordinary random subset.
- One YAML file per experiment, under `configs/experiments/`. Do not invent ad-hoc config names elsewhere.
- Write checkpoints only under `cs604-outputs`. Do not commit checkpoints, archives, or prediction dumps.
- Do not start a real or long training run unless the user asks for that run. A code change is verified with the smoke test and unit tests.
- The desktop GPU runs one training job at a time. Do not launch `scripts/train.py` on real data while another job is using the GPU.
- In Colab, stage images on `/content`. Do not train directly from the Drive image tree.
- Edit only the files the current task names. Leave unrelated dirty files alone.

## Layout

```
configs/                  experiment settings
configs/experiments/      one config per run
src/cs604/                data, models, training, prediction
scripts/                  train, predict, prototypes, dataset checks
tests/                    unit tests
notebooks/                Colab launcher
outputs/submissions/      audited CodaBench files
report/                   experiment record and final report
```

`scripts/train.py` is the local and Colab entry point. Model modes live in `src/cs604/models/build.py`: full fine-tune, frozen backbone, last blocks, and LoRA on attention `qkv`.

## Verify

Smoke test, no dataset required:

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\smoke.yaml --synthetic
```

Unit tests:

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" -m unittest tests.test_pipeline
```

A real local run, only when the user asks:

```powershell
& "C:\Users\Van Le\.venvs\cs604\Scripts\python.exe" scripts\train.py --config configs\local.yaml
```

Resume with `--resume C:\Users\Van Le\cs604-project\cs604-outputs\local-full\last.pt` and the same config. Every run should keep its resolved config, git commit, `hardware.json`, parameter counts, timing, and metrics.

## Submissions

One line per image: `image_path class_id`, with `class_id` in `[0, 9999]`. Preserve `val.txt` / `test.txt` order. Name the file `val_team_xx.txt` or `test_team_xx.txt`. Generate it with `scripts/predict.py`, which checks row count, paths, and class range.

## Report numbers

Copy metrics from `run.json`, `metrics.csv`, and the `.inference.json` beside each prediction. Record them in `report/EXPERIMENTS.md`. The index is `report/experiment_index.csv`, rebuilt by `scripts/summarize_experiments.py`. Do not reconstruct numbers from memory at report time.
