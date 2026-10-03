# CS604 agent instructions

Fine-grained 10,000-class classification with a ViT-B-sized vision transformer (currently ViT-B/16). The course requires a full fine-tune and a parameter-efficient method, under a 2-hour × 48 GB training budget.

## Course rules confirmed by the Prof (2026-10-03)

These override anything older in `report/`, `README.md`, or `.cursor/plans/`.

1. **Train the classifier head in every experiment.** The Prof expects head
   retraining. New configs set `train_head: true` and start the head from the
   class prototypes (`prototype_file`). A frozen head (`train_head: false`) is
   allowed only as a labelled ablation, with a comment at the top of the YAML
   saying so.
2. **The head is not counted in the 5% limit.** "Under 5%" applies to the
   backbone's trainable parameters (LoRA, adapters, biases, selected blocks).
   The backbone allowance is therefore about 4.3M parameters for ViT-B (rank-8
   `qkv` LoRA uses 0.29M). Report trainable % both without the head (the course
   number: `backbone_trainable_params / total_params`) and with it.
   `run.json`'s `trainable_pct` includes the head.
3. **Any vision transformer of comparable size (about 86M, ViT-B class) may be
   the backbone**, not only `vit_base_patch16_224.augreg2_in21k_ft_in1k`.
   Candidates: DINOv2 ViT-B/14, CLIP/SigLIP ViT-B/16 image encoders, EVA-02
   ViT-B, `vit_base_patch16_384`. Ask before using non-ViT or hybrid models
   (ConvNeXt, Swin).
4. **No checkpoints trained or fine-tuned on iNaturalist 2021 or TreeOfLife-10M**
   (for example BioCLIP, iNat-finetuned timm weights). The course data appears to
   be iNat2021 "train_mini" (see `report/INAT2021_STRATEGY.md`), so its test
   images may be in their training data. Check each model's pretraining data
   and record it in the report.
5. Earlier frozen-head results (43.90% prototypes, 51.094% screen, 51.826% and
   52.924% LoRA runs) are now the **frozen-head ablation**, not the method.

Current status and the next experiments are in `report/HANDOFF.md`; the
experiment order is in `report/INAT2021_STRATEGY.md`.

## Paths

This git repo is the `prototype-baseline` worktree,
`C:\Users\Van Le\cs604-project\cs604-prototype-baseline`. All agents work here.
`C:\Users\Van Le\cs604-project\cs604-group-project` is the original checkout
on `main` and is out of date.

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

`scripts/train.py` is the local and Colab entry point. Model modes live in `src/cs604/models/build.py`: full fine-tune, frozen backbone, last blocks, and LoRA on attention `qkv`. `train_head` controls whether the classifier trains in `lora` and `last_blocks` modes; `full` and `head` always train it.

Not implemented yet (needed by the next experiments): a separate head learning
rate, a learnable cosine temperature while the head is frozen, a configurable
train-crop scale and eval crop, configurable input resolution, and backbones
other than timm `vit_base_patch16_*`. Until a feature exists, do not put its
key in a config: unknown keys are silently ignored.

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

## Workflow

`WORKFLOW.md` is the runbook from config to recorded CodaBench score. Follow it, and update it when a script or step changes.

## Submissions

One line per image: `image_path class_id`, with `class_id` in `[0, 9999]`. Preserve `val.txt` / `test.txt` order. Name the file `val_team_xx.txt` or `test_team_xx.txt`. Generate it with `scripts/predict.py`, which checks row count, paths, and class range.

## Report numbers

Copy metrics from `run.json`, `metrics.csv`, and the `.inference.json` beside each prediction. Record them in `report/EXPERIMENTS.md`. The index is `report/experiment_index.csv`, rebuilt by `scripts/summarize_experiments.py`. Do not reconstruct numbers from memory at report time.
