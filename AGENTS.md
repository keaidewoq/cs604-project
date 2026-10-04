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
   the backbone**, not only the course model (see rule 6 for where to start).
   Candidates: DINOv2 ViT-B/14, CLIP/SigLIP ViT-B/16 image encoders, EVA-02
   ViT-B, `vit_base_patch16_384`. Ask before using non-ViT or hybrid models
   (ConvNeXt, Swin).
4. **No checkpoints trained or fine-tuned on iNaturalist 2021 or TreeOfLife-10M**
   (for example BioCLIP, iNat-finetuned timm weights). The course data appears to
   be iNat2021 "train_mini", so its test
   images may be in their training data. Check each model's pretraining data
   and record it in the report.
5. Earlier frozen-head results (43.90% prototypes, 51.094% screen, 51.826% and
   52.924% LoRA runs) are now the **frozen-head ablation**, not the method.
6. **Start from the ImageNet-21k-only ViT-B/16, not the IN21k→IN1k fine-tune.**
   The Prof thinks the IN21k weights are better. This matches the PDF's
   constraint line, "Base Model – vit-base-patch16-224-in21k". In timm:
   - `vit_base_patch16_224.orig_in21k`: the original Google weights, the same
     as HF `google/vit-base-patch16-224-in21k` named in the PDF.
   - `vit_base_patch16_224.augreg_in21k`: the newer "How to train your ViT"
     IN21k weights.
   Build prototypes for both and pick by prototype accuracy on the fixed
   holdout; report which one is used. Every existing prototype file, step-zero
   gate (43.90%) and result belongs to `augreg2_in21k_ft_in1k`. A new backbone
   needs its own prototype file and its own measured `expected_initial_accuracy`
   (from an evaluate-only run). Never pair prototypes with a different backbone.

Results are in `report/EXPERIMENTS.md` and `report/progress_log.md`. If they
exist locally (they are gitignored), `report/HANDOFF.md` holds the current
status and next experiments, and `report/INAT2021_STRATEGY.md` the experiment order.

## Paths

Machine-specific paths (this machine's Python interpreter, worktree, GPU)
live in `AGENTS.local.md`, which is gitignored. Read it first if it exists.

| What | Where |
| --- | --- |
| Python (CUDA) | a virtualenv with a CUDA build of PyTorch (see `AGENTS.local.md`) |
| Dataset | `${CS604_DATA}`, default `../cs604-course-data` beside the repo |
| Checkpoints and logs | `${CS604_OUTPUTS}`, default `../cs604-outputs` beside the repo |
| CodaBench files | `${CS604_OUTPUTS}/<EXPERIMENT>/submissions/` |
| Run records for git | `outputs/<EXPERIMENT>/`, copied by `scripts/publish_run.py` (see `outputs/README.md`) |

Configs write data and output paths as `${CS604_DATA}/...` and
`${CS604_OUTPUTS}/...`; `src/cs604/config.py` expands them. Never put an
absolute, machine-specific path in a committed file. A CPU-only `python`
cannot see the GPU, so always call the CUDA virtualenv's interpreter. Run
commands from the repo root.

## Hard rules

- Validation and test labels are withheld. Do not hand-label those images, train on them, or infer labels from filenames.
- Keep the internal split fixed: the same five images per class. Screening configs use fewer training images per class (5, then 10, then 20). Never replace that with an ordinary random subset.
- One YAML file per experiment, under `configs/experiments/`. Do not invent ad-hoc config names elsewhere.
- Write checkpoints only under `cs604-outputs`. Do not commit checkpoints, archives, or prediction dumps. The repo's `outputs/` takes only the small records `publish_run.py` copies; large files are shared through the team Drive folder `CS604_RUNS`.
- Do not start a real or long training run unless the user asks for that run. A code change is verified with the smoke test and unit tests.
- A local GPU runs one training job at a time. Do not launch `scripts/train.py` on real data while another job is using the GPU.
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
outputs/<EXPERIMENT>/     run records (run.json, metrics.csv, ...)
report/                   experiment record and final report
```

`scripts/train.py` is the local and Colab entry point. Model modes live in `src/cs604/models/build.py`: full fine-tune, frozen backbone, last blocks, and LoRA on attention `qkv`. `train_head` controls whether the classifier trains in `lora` and `last_blocks` modes; `full` and `head` always train it.

Optional training keys: `head_lr` gives the trainable classifier (cosine weight
and temperature) its own learning rate, needs `train_head: true`, and follows
the same warmup/cosine schedule; unset, the head shares `lr`.
`train_crop_scale: [min, max]` sets the RandomResizedCrop area range for
training; unset, timm's default `[0.08, 1.0]` applies. `run.json` records `lr`,
`head_lr`, and the `train_crop_scale` the train transform actually used.
`cosine_scale` (default 20) is the cosine head's initial temperature;
`cosine_scale_trainable: false` keeps it fixed (an ablation; rule 1 trains it).
`linear_head_fp32: true` keeps a plain linear head's logits float32 under bf16
autocast (the linear twin of `cosine_head_fp32`). All three are recorded in `run.json`.
`image_size` is passed to timm as `img_size`, which resizes the position
embeddings; the transforms follow the built model (`model_data_config`), with
the checkpoint's own mean/std and crop. Any timm `VisionTransformer` with a
`qkv` Linear works, e.g. DINOv2 `vit_base_patch14_dinov2.lvd142m` at
`image_size: 224` (default 518; at 224 it has 256 patches versus 196 for
ViT-B/16, about 30% more compute per image).

Not implemented yet (needed by the next experiments): a learnable cosine
temperature while the head is frozen, a configurable eval crop, and timm model
classes other than `VisionTransformer` (EVA-02 uses `Eva`; CLIP/SigLIP image
towers are untested). Until a feature exists, do
not put its key in a config: unknown keys are silently ignored.

## Verify

Smoke test, no dataset required:

```powershell
python scripts\train.py --config configs\smoke.yaml --synthetic
```

Unit tests:

```powershell
python -m unittest discover -s tests
```

A real local run, only when the user asks:

```powershell
python scripts\train.py --config configs\local.yaml
```

Resume with `--resume ../cs604-outputs/local-full/last.pt` and the same config. Every run should keep its resolved config, git commit, `hardware.json`, parameter counts, timing, and metrics.

## Workflow

`WORKFLOW.md` is the runbook from config to recorded CodaBench score. Follow it, and update it when a script or step changes.

## Submissions

One line per image: `image_path class_id`, with `class_id` in `[0, 9999]`. Preserve `val.txt` / `test.txt` order. Name the file `val_team_xx.txt` or `test_team_xx.txt`. Generate it with `scripts/predict.py`, which checks row count, paths, and class range.

## Report numbers

Copy metrics from `run.json`, `metrics.csv`, and the `.inference.json` beside each prediction. Record them in `report/EXPERIMENTS.md`. The index is `report/experiment_index.csv`, rebuilt by `scripts/summarize_experiments.py`. Do not reconstruct numbers from memory at report time.
