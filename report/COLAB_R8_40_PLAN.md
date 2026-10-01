# Colab rank-8 finalist: 40 training images per class

Status: implementation plan, 2026-10-01. This file does not authorize or record a
training run. Implement the preparation in Cursor, then run one Colab job after the
preflight gates below pass.

## Goal and evidence

Train a fresh ViT-B/16 with attention `qkv` LoRA rank 8 and the frozen cosine
classifier initialized from the existing 40-image-per-class prototypes. Reserve
the same five labeled images per class as the internal holdout. Train on all
remaining 40 images per class (400,000 images, 12,500 updates at batch size 32
for one epoch). Measure step zero and the final model on the 50,000-image fixed
holdout. Keep the better checkpoint. Generate predictions for the separate,
official `val` split on the laptop, then submit the validated zip to CodaBench.

The local rank-8 screen is the reference, not the initializer: step zero was
43.900%; after two epochs over ten training images per class it reached 51.094%.
Its source is `cs604-outputs/prototype-lora-r8-fp32-screen/{baseline.json,
metrics.csv,run.json}`. The new run starts from `prototypes/class_prototypes.pt`,
not from the screen's trained LoRA checkpoint. The prototype source used 40
training images per class; the official validation images have never been used
for fitting. One epoch over 40 images and two epochs over ten images are different
training schedules, so a score difference cannot be attributed to sample count
alone.

## 1. Prepare and pin the inputs

1. Add one new YAML under `configs/experiments/` and give it a new Drive output
   directory. Keep `mode: lora`, rank 8, alpha 16, dropout 0.05, `head_type:
   cosine`, `train_head: false`, `cosine_head_fp32: true`, bf16, batch 32, seed 0,
   split seed 604, `val_per_class: 5`, `train_per_class: null`, and one epoch.
   Keep the screen's LR 0.0005, weight decay 0.01, and label smoothing 0 unless
   a separately documented decision changes them. Leave `max_steps: null` so
   the intended run is one complete epoch. Use a Colab path for `prototype_file`
   and `baseline_reference_split`; the notebook CLI only overrides `data_root`
   and `output_dir`.
2. Require `evaluate_before_training: true`, expected initial accuracy 0.439,
   tolerance 0.001 (fractions, equal to 0.10 percentage points), and exactly
   50,000 initial validation samples. Preserve `validate: true`. Do not change
   the gate to accommodate an unexpected score.
3. Store copies of `cs604-outputs/prototypes/class_prototypes.pt` and
   `cs604-outputs/prototypes/splits/internal_val.txt` in a stable Drive input
   directory. Record their SHA-256 hashes before upload and verify the Drive
   copies. The original prototype hash recorded by the prior baseline is
   `c4c90ef9443c1430c519c4496829117563e21c796037f764a89c5ab1ea11b68f`;
   the holdout hash is
   `19a10c37a775559f4eee516215ad10c4b906ee50d66f7cbd95bc5ec88f2f2f69`.
   Copy these inputs from Drive to `/content` at the start of each Colab session
   and verify the copies there too. Do not rebuild or overwrite the originals.
4. Commit the config, notebook, and checkpoint-recovery implementation on
   `prototype-baseline`, then push that branch to the repository used by Colab.
   Pin the notebook's `COMMIT` to the resulting full SHA, not `main` or the
   earlier `f6c5e3c`. Verify the checkout's HEAD and clean state in Colab.
   Do not commit checkpoints, archives, or predictions.

## 2. Make checkpoints survive a Colab interruption

The current `_save_training_checkpoint` writes directly over `last.pt`. Replacing
that file on a Drive mount can corrupt the only recent resume point. Implement
and test this protocol before the real run:

1. Serialize each checkpoint first to a temporary file on the VM's `/content`
   disk. Load it back with `torch.load` and verify the expected model, optimizer,
   scheduler, config, step, epoch, and next-batch fields.
2. Copy each periodic resume checkpoint to a *new, step-numbered* path in the
   run's Drive directory. Load the Drive copy and verify its step and checksum
   before marking it usable. Keep at least the previous verified checkpoint;
   never rely on a rename within the Drive mount or delete the old copy before
   the new one has passed read-back. A partial new copy must be ignored on
   recovery. Retain a usable `last.pt` for the normal interface, but preserve
   the numbered copies as its fallback. The `--resume` argument may point
   directly to a verified numbered checkpoint.
3. Protect `baseline.pt` and `best.pt` the same way, including the final
   replacement of the step-zero `best.pt`. Keep a verified versioned copy of
   each. The baseline files are required for a protected resume. The model
   needs a fresh output directory if the initial accuracy gate never completes
   and no valid `baseline.json`/`best.pt` pair exists; retain that incomplete
   directory for diagnosis.
4. Keep `metrics.csv`, `run.json`, resolved `config.json`, and checkpoint hashes
   in Drive. Checkpoint writes should not silently succeed only on `/content`.
   Use a periodic interval that bounds lost work without excessive Drive I/O;
   500-1,000 updates is a starting range, to be checked in the Colab preflight.

Before the long run, perform a short Colab write/read-back and interrupted-resume
test in a separate smoke directory. Verify that model, optimizer, scheduler,
step, and next-batch state restore, that the same output directory is used, and
that an intentionally incomplete newest copy falls back to the preceding valid
checkpoint. The code resumes the seeded batch order but does not save every RNG
state; a resumed run need not be bit-for-bit identical to an uninterrupted run.

## 3. Colab preflight and one real run

1. Select the intended Colab GPU and record its actual name and VRAM with
   `nvidia-smi`; do not assume a particular device from the runtime label.
   Confirm CUDA PyTorch, dependencies, enough `/content` space for the staged
   archives and extracted images, enough Drive space for checkpoints, and the
   expected 400,000/50,000 split counts. `scripts/stage_data.py` stages the
   train and official val archives; training must read images from `/content`,
   never from the Drive image tree.
2. Confirm the pinned commit, source hashes, new empty output directory, and
   that this is the only GPU training job. Launch `scripts/train.py` with the
   new YAML, `/content/cs604-course-data` as data root, and the unique Drive
   output directory. Do not reuse the screen or validation directories.
3. Stop before optimizer updates unless `baseline.json` says `passed: true`,
   step zero is within 43.80%-44.00%, the holdout has 50,000 images, and its
   SHA-256 equals the saved reference. Investigate any mismatch in input
   files, transforms, precision, or pinned code rather than widening the gate.
4. At periodic checkpoints, confirm a verified Drive resume copy exists.
   During the single epoch `best.pt` remains the zero-update model until final
   holdout evaluation; use a verified `last.pt` or numbered resume copy if the
   session ends early. On a new runtime, restage inputs, check the same pinned
   commit and config, verify the selected checkpoint, and resume in the *same*
   output directory with `--resume <verified checkpoint>`. Do not delete the
   partial run or start a new one over its files. If the latest copy is damaged,
   use the preceding verified checkpoint.
5. After completion, require `run.json` status `completed`, 12,500 optimizer
   updates, an initial and final validation row in `metrics.csv`, reloadable
   `baseline.pt`, `best.pt`, and `last.pt`, and a recorded final accuracy and
   loss. Confirm whether `best.pt` is step zero or the trained model; only a
   strictly better internal score should replace step zero. Record GPU name,
   logged elapsed time, peak VRAM, and normalized 48-GiB hours. If a run resumes,
   preserve each session's elapsed time before `run.json` is rewritten; its final
   `elapsed_sec` covers only the last process, and the `metrics.csv` elapsed
   clock restarts. Sum the GPU time across sessions for the budget report.
   The course budget is 2 normalized 48-GiB hours; estimate from measured Colab
   throughput before committing to the long run, and report actual usage
   afterward.

## 4. Bring the result home and submit

1. Download the verified `best.pt` from Drive to a new local output directory,
   along with `run.json`, `baseline.json`, `metrics.csv`, resolved config, and
   hash manifest. Compare the local checkpoint SHA-256 with the Drive copy and
   load the local checkpoint. Keep the existing local screen and prototype
   artifacts untouched.
2. On the laptop, use the CUDA virtualenv and `scripts/predict.py` with that
   `best.pt`, the local course dataset, `--split val`, and a filename of the
   form `outputs/submissions/val_team_xx.txt` (replace `xx` with the real team
   identifier). The checkpoint contains model weights and config, so local
   prediction does not require the Colab prototype file path to exist. The
   script checks `val.txt` order, row count, and class range and writes a zip
   and `.inference.json` beside the text file.
3. Check the generated file has 50,000 rows and preserve the zip, text file,
   inference record, checkpoint hash, and run metrics locally. Submit the
   audited zip to the Phase 1 CodaBench validation leaderboard. Record its
   public score separately from the internal holdout result; never treat the
   official validation images as labeled training data.
4. Add the measured result, paths, commit, timing, and public score (when
   available) to `report/EXPERIMENTS.md`, `report/progress_log.md`, and
   `outputs/experiment_index.csv`. State that the classifier prototypes used
   40 training images per class and the LoRA run also trained on the 40
   non-holdout images per class.

## Completion criteria

The workflow is complete when the exact pinned code and hashed inputs produced
a recoverable Colab run, the fixed internal holdout was measured at step zero
and after training, the selected checkpoint was verified on both Drive and the
laptop, local prediction passed the submission checks, and the public score and
resource usage were documented. Until then, preserve every earlier checkpoint
and keep the new run in its own directory.
