# Experiment workflow

The routine for one experiment, from a local screen to a recorded CodaBench
score. Every experiment has one name, `EXPERIMENT`, and that name appears
everywhere:

| What | Where |
| --- | --- |
| Settings | `configs/experiments/<EXPERIMENT>.yaml` (git) |
| Colab output | `My Drive/cs604/outputs/<EXPERIMENT>/` (Google Drive for desktop shows it locally, e.g. `H:\My Drive\cs604\outputs\<EXPERIMENT>`) |
| Permanent copy | `../cs604-outputs/<EXPERIMENT>/` (or `$CS604_OUTPUTS`) |
| Submission files | `cs604-outputs\<EXPERIMENT>\submissions\` |
| CodaBench ID and score | `cs604-outputs\<EXPERIMENT>\leaderboard.json` |
| One row per run | `report/experiment_index.csv` (git) |

Name experiments so they sort and read well, for example
`e012_lora_r16_lr3e-4_screen10` for a local screen and `e012_lora_r16_lr3e-4_40`
for its full Colab run.

**Overview:**

1. Set up a terminal (section 0).
2. Screen the idea locally (section 1).
3. Promote it to a full Colab run (section 2).
4. Copy the run to the laptop (section 3).
5. Predict and submit (section 4).
6. Record the CodaBench result (section 5).

## 0. Terminal and virtual environment

Use a virtualenv with a CUDA build of PyTorch; a CPU-only `python` cannot see
the GPU. Your own interpreter path belongs in `AGENTS.local.md` (gitignored),
not in this file.

**PowerShell:**

```powershell
cd <path-to-this-repo>
& <path-to-venv>\Scripts\Activate.ps1
```

If PowerShell refuses to run `Activate.ps1`, allow scripts for this window only
with `Set-ExecutionPolicy -Scope Process Bypass`, then run the activate line
again.

**Git Bash:**

```bash
cd <path-to-this-repo>
source <path-to-venv>/Scripts/activate
```

**Check the GPU is visible.** This should print `True` and your GPU's name:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The `python` commands below assume the virtualenv is active and the current
folder is the repo root. Data and outputs are the sibling folders
`../cs604-course-data` and `../cs604-outputs`, or wherever `CS604_DATA` and
`CS604_OUTPUTS` point.

**Code and branch:** work on your own branch, which is the branch Colab checks
out (`BRANCH` in the notebook's checkout cell). Run `git status` before you
start; commit or discard leftovers so `run.json` records `git_dirty: false`.

## 1. Screen an idea on the laptop

Screens use fewer training images per class (5, then 10, then 20) and the fixed
50,000-image internal holdout (5 images per class, `split_seed: 604`). Never
change the holdout or replace it with a random subset.

1. **Write the config.** Copy the closest existing one, for example
   `configs/experiments/prototype_lora_r8_trainhead_screen.yaml`, to
   `configs/experiments/<EXPERIMENT>.yaml`. Change only what the experiment
   tests, plus `train_per_class` and `epochs`. Write a comment at the top saying
   what it tests and which screen it is compared against. Keep the paths as
   `${CS604_DATA}/...` and `${CS604_OUTPUTS}/...`; never an absolute path.
   - **Train the head** (`train_head: true`, prototype-initialized) in every
     new config. Most older configs have `train_head: false`; copying one
     means changing that line. A frozen head is only for a labelled ablation.
     See "Course rules" in `AGENTS.md`.
   - `head_lr` sets the head's learning rate (cosine weight and temperature);
     unset, the head uses `lr`. It requires `train_head: true`.
   - `train_crop_scale: [min, max]` sets the training RandomResizedCrop area
     range; unset, timm's `[0.08, 1.0]`. Both values are recorded in `run.json`.
   - Do not add keys for features that are not implemented yet (listed in
     `AGENTS.md`); unknown keys are ignored without warning.
   - **A new backbone needs its own prototypes and gate** (rule 6 in
     `AGENTS.md`). Build them with
     `python scripts/build_prototypes.py --config configs/experiments/prototype_init_<variant>.yaml`
     (output from the config's `prototype_output`; an existing build is never
     overwritten), then run its evaluate-only `prototype_validate_<variant>.yaml`
     with `expected_initial_accuracy: null`. Use the measured accuracy as
     `expected_initial_accuracy` in that backbone's training configs. Training
     refuses prototypes built with a different `timm_checkpoint`.
2. **Make sure the GPU is free.** Only one training job at a time:
   ```powershell
   nvidia-smi
   ```
3. **Smoke-test only if you changed code** in `src/` or `scripts/`. Run the
   unit tests and a 2-step synthetic LoRA run (about 1 minute):
   ```powershell
   python -m unittest tests.test_budget tests.test_pipeline tests.test_checkpoint_io tests.test_baseline tests.test_stage_data tests.test_workflow
   python scripts/train.py --config configs/experiments/colab_checkpoint_smoke.yaml --synthetic --max-steps 2 --output-dir "../cs604-outputs/smoke-check"
   ```
   Delete `cs604-outputs\smoke-check` afterwards. A config-only change needs no
   smoke test. Configs with `evaluate_before_training: true` refuse synthetic
   data, and the real run checks the 43.90% starting score in its first few
   minutes anyway.
4. **Commit the config before the real run**, so the run records a clean commit:
   ```powershell
   git add configs/experiments/<EXPERIMENT>.yaml
   git commit -m "Add <EXPERIMENT> screen"
   git push origin <your-branch>
   ```
5. **Run the screen.** The output folder is named after the experiment:
   ```powershell
   python scripts/train.py --config configs/experiments/<EXPERIMENT>.yaml --output-dir "../cs604-outputs/<EXPERIMENT>"
   ```
   - Prototype-based configs first check that the step-zero score is
     43.90% ± 0.10. If it isn't, the run stops before training.
   - The laptop is slow, about 30 images/s while training. A 10-image,
     two-epoch screen takes about 2 hours.
   - The run refuses to write into a folder that already has a baseline.
6. **Resume after an interruption** with the same command plus
   `--resume "../cs604-outputs/<EXPERIMENT>/checkpoints/resume_step_<N>.pt"`.
   If the newest numbered file is damaged, use the one before it.
7. **Read the result** in `cs604-outputs\<EXPERIMENT>\run.json`
   (`best_internal_accuracy`) and `metrics.csv`. Add the run to the index
   (section 5, step 3) and write a short entry in `report/progress_log.md`.

Promote an idea to Colab only when its screen beats the current best screen.

### 1b. Head-only sweeps on cached features

With a frozen backbone (`mode: head`) the head only sees each image's
pre-logits feature, so head settings can be swept on cached features in about
2 minutes per 10-epoch setting instead of hours end to end.

1. **Cache the features** (GPU, about one pass over the data per view):
   ```powershell
   python scripts/cache_features.py --config configs/experiments/feature_cache_orig_in21k.yaml --groups holdout clean aug_default
   ```
   The config lists view groups: the holdout and the training images with the
   eval transform, and seeded augmented views made with the training transform
   (`build_transforms`, the one `train.py` uses). The split must match the
   prototype build's split files byte for byte. Finished views are skipped, so
   rerun the same command after an interruption. Output and `manifest.json`
   (backbone, split hashes, transforms, view seeds, shapes, hashes) go under
   `cs604-outputs/feature_cache/`, never in git. About 0.6 GB per train view.
2. **Run the sweep:** one YAML (`base` config plus named `points`), results in
   `<output_dir>/sweep.csv` and `points/<name>.json`:
   ```powershell
   python scripts/sweep_cached_head.py --config configs/experiments/head_sweep_orig_in21k.yaml
   ```
   Finished points are skipped; add new rounds as new points, never edit a
   finished one. Each point checks the step-zero gate on the cached holdout.
3. **Confirm end to end.** Cached results are a screen: run the best settings
   as ordinary `mode: head` experiment YAMLs (section 1) before promoting one.

## 2. Full training on Colab

The Colab VM (RTX PRO 6000, 95 GB) trains one epoch over all 40 images per
class in about 7 minutes. Staging the data takes 20–40 minutes, once per runtime.

**Compute units:**
- Colab bills every minute a runtime is connected, at a rate set by the GPU
  type, whether or not anything is running.
- On 2026-10-01 the whole monthly allowance (100 units) went in one day. About
  5 hours were spent connected on the RTX PRO 6000, mostly debugging data
  staging; training itself was 7 minutes.
- Before connecting, read the rate under Colab's RAM/disk indicator → View resources.
- A LoRA or head run needs only about 2 GB of GPU memory, so the cheapest GPU
  that finishes in acceptable time is the right choice.
- Never debug on a GPU runtime. Debug on the laptop.
- Queue several experiments per session, so one data staging covers all of them.
- Check colab.research.google.com → Runtime → Manage sessions for forgotten
  runtimes. Closing Cursor does not end one.

1. **Write the full-run config.** Copy the screen config to
   `configs/experiments/<EXPERIMENT>.yaml` under the full-run name, for example
   `..._40`. Change:
   - `train_per_class: null` (all 40 training images per class) and `epochs` as wanted.
   - The paths to Colab paths, using
     `configs/experiments/prototype_lora_r8_fp32_40.yaml` as the template:
     ```yaml
     data_root: "/content/cs604-course-data"
     output_dir: "/content/drive/MyDrive/cs604/outputs/<EXPERIMENT>"
     prototype_file: "/content/cs604-inputs/class_prototypes.pt"
     baseline_reference_split: "/content/cs604-inputs/internal_val.txt"
     ```
     The notebook passes `data_root` and `output_dir` itself. It passes
     `prototype_file` only if the config has one. `baseline_reference_split`
     must be the `/content` path. Its file is byte-identical to the laptop's
     `prototypes/splits/internal_val.txt`, so Colab and laptop scores are
     comparable.
   - **Time budget.** To use the course budget, set `budget_48gb_hours`
     (for example `1.9`, leaving room under the 2.0 + 10% limit) and make
     `epochs` a generous upper bound. Template:
     `configs/experiments/prototype_lora_r8_fp32_budget_40.yaml`.
     - The run converts the budget to wall-clock time for the GPU it gets:
       2.0 × 48 GB-hours is 1.0 h on the 96 GB RTX PRO 6000, about 2.4 h on
       an A100 40 GB, and 1.2 h on an A100 80 GB.
     - The cosine learning-rate schedule follows whichever is further along,
       steps or time, so it decays to zero by the deadline. Training then
       stops, leaving time for the final holdout evaluation.
     - Each log line shows the share of the budget used and how many epochs
       will fit. `run.json` records `budget_wall_sec` and `stopped_by_budget`.
     - Elapsed time is stored in every checkpoint, so a resumed run continues
       the same clock. Time after the last checkpoint is lost and not counted.
     - The clock starts after the model and data index are built, and it
       includes every holdout evaluation.
   - `ckpt_every`: 1000 for runs of an hour or more, so a disconnect loses
     little work. For short runs it can be larger, for example 5000; a
     checkpoint is also written at the end of each epoch.
2. **Commit and push it:**
   ```powershell
   git add configs/experiments/<EXPERIMENT>.yaml
   git commit -m "Add <EXPERIMENT> Colab run"
   git push origin <your-branch>
   ```
   Colab checks out the branch tip, so an unpushed config cannot run.
3. **Check Drive space.** Keep at least 2 GB free in `My Drive`. With compact
   checkpoints (about 34 MB each), a LoRA or head run needs well under 1 GB.
   Empty the Drive trash after deleting anything.
4. **Open `notebooks/colab_train.ipynb` in Cursor.** Use Select Kernel → Colab
   → GPU, signed in with the Google account that has your Colab plan. Reload the notebook
   if it changed in git.
5. **Run the cells in order** (table below), then read the notes after it.

| Cell | What it does | Expect |
| --- | --- | --- |
| GPU check | `nvidia-smi`; defines `run_logged` | GPU listed. Re-run it after any kernel restart. |
| Mount Drive | mounts `/content/drive` | "Mounted" or "already mounted" |
| GitHub token | reads `cs604/secrets/github_token.txt` | "GitHub token ready" |
| Checkout | clones or fetches the `BRANCH` tip; installs requirements | `code ready ... commit <sha>`: your latest push |
| Inputs | copies `class_prototypes.pt` and `internal_val.txt` from Drive and checks their hashes | both "ok" |
| Stage data | streams `train.tar.gz` into `/content`; fetches `train.txt` | `data ok ... train.txt lines: 450000` |
| Debug cells (if present) | disk and HF access checks | only if stage data fails |
| Checkpoint smoke | 4 synthetic steps, corrupts a checkpoint, resumes | `colab checkpoint smoke passed` |
| Training | set `EXPERIMENTS = ["<EXPERIMENT>", ...]`, then run | `===== <name> =====` blocks with `step ... img/s` and `validation top1=...`, a `summary`, then the runtime ends |
| Disconnect (last cell) | flushes Drive, then ends the runtime now | use it whenever you stop working without auto-disconnect |

   - **Stage data** reuses images already on the VM. If it stops on low disk,
     run the debug disk cell. Cursor doesn't show `!` shell output, so use
     `subprocess.run(..., capture_output=True)` and print the result. Free
     space with Runtime → Disconnect and delete runtime, or delete
     `/content/cs604-course-data`, then re-run.
   - **Training** runs every name in `EXPERIMENTS` back to back on the staged
     data.
     - Before anything trains, it checks every config is in the checkout and
       no output folder already holds a run. If a check fails, fix it and
       re-run the cell; the runtime stays connected.
     - A failed experiment doesn't stop the others.
     - Each run's console log is saved as `train.log` in its Drive folder.
   - **Auto-disconnect:** with `AUTO_DISCONNECT = True` (the default), the cell
     ends the runtime once training has run, even if a run failed, so it stops
     using units.
     - It does not disconnect after a check failure or a manual interrupt.
       Then it prints a reminder; run the Disconnect cell when you're done.
     - Before ending the runtime it runs `drive.flush_and_unmount()`, so the
       last checkpoints, `run.json`, and `train.log` finish uploading to Drive.
     - **Watchdog:** once the checks pass, a timer ends the runtime after a
       hard limit even if training hangs and never returns. With
       `WATCHDOG_HOURS = None` the limit is each config's
       `budget_48gb_hours` in wall-clock time on this GPU, plus 30 minutes per
       experiment (about 2.8 h for one 1.9-hour-budget run on an A100 40 GB).
       Configs without a budget need `WATCHDOG_HOURS` set to a number. After
       an interrupt the watchdog stays armed; `WATCHDOG.cancel()` stops it.
     - After a run, confirm on colab.research.google.com → Runtime → Manage
       sessions that nothing is still active.
   - **Resume** an interrupted run with
     `RESUME = {"<name>": ".../checkpoints/resume_step_XXXXXX.pt"}`, using the
     newest good checkpoint on Drive.
6. **If the runtime disconnects:** `/content` is wiped but Drive is not.
   Re-run from the top. Stage data downloads again, and training resumes
   with `RESUME`.
7. Delete `My Drive/cs604/outputs/colab-checkpoint-smoke` once it has passed.

## 3. Copy the run to the laptop

Google Drive for desktop shows My Drive as a drive letter (`H:\My Drive` by
default in `sync_run.py`; set `CS604_DRIVE_OUTPUTS` or `--drive-root` if yours differs). When the training
cell prints `artifacts`, wait for Drive to finish syncing, then:

```powershell
python scripts/sync_run.py <EXPERIMENT> --prune-drive
```

- Copies the whole run folder to `cs604-outputs\<EXPERIMENT>\`, except the
  numbered `checkpoints\*_step_*.pt` copies.
- Checks every file by SHA-256. `last.pt`, `best.pt`, and `baseline.pt` must
  also match the hashes training logged, which catches a half-synced Drive file.
- Refuses a run that has not finished, and never overwrites a local copy that
  differs. If it says Drive may still be syncing, wait and run it again.
- `--prune-drive` then deletes the numbered checkpoints on Drive. Empty the
  Drive trash afterwards.
- Once you're happy with the local copy, you may delete the whole Drive run folder.

Use another Drive location with `--drive-root` or the `CS604_DRIVE_OUTPUTS`
environment variable.

## 4. Predict and make the CodaBench file

Run this on the laptop, with no training job using the GPU. The 50,000
validation images take about 6 minutes.

```powershell
python scripts/predict.py --checkpoint "../cs604-outputs/<EXPERIMENT>/best.pt" --data-root "../cs604-course-data" --split val --output "../cs604-outputs/<EXPERIMENT>/submissions/val_team_xx.txt"
```

- Replace `xx` with the team number. In Phase 2, use `--split test` and
  `test_team_xx.txt` once the test images and `test.txt` are in `cs604-course-data`.
- `best.pt` is the checkpoint with the best internal holdout score. It is the
  step-zero model if training never beat the starting score.
- `predict.py` checks the row count, image order, and class range before
  writing. It writes three files:
  - `val_team_xx.txt`: the predictions.
  - `val_team_xx.zip`: **the file to upload**.
  - `val_team_xx.inference.json`: checkpoint and file hashes, the git commit, and the GPU.
- Laptop predictions can differ from the Colab holdout evaluation by a few
  hundredths of a point because of GPU arithmetic. The submitted file is the
  laptop's.
- Do not edit the `.txt` or `.zip` after this step. Recording the submission
  checks their hashes.

## 5. Submit and record the CodaBench result

1. **Upload `val_team_xx.zip` to CodaBench.** Note the submission ID it shows.
2. **Record it straight away**, even before the score appears:
   ```powershell
   python scripts/record_submission.py <EXPERIMENT> --submission "../cs604-outputs/<EXPERIMENT>/submissions/val_team_xx.zip" --submission-id <ID> --phase val
   ```
   When the score is out, run the same command with `--score 0.5123` added.
   Use the accuracy as a fraction: 51.23% is 0.5123. Running it again with the
   same ID updates that entry.
   - The command checks that the uploaded file is unchanged since `predict.py`,
     and that the checkpoint behind it is in this run folder.
   - It writes `leaderboard.json` in the run folder. A run can have several
     submissions; the best score becomes the run's `public_accuracy`.
3. **Rebuild the index and commit it:**
   ```powershell
   python scripts/summarize_experiments.py
   git add report/experiment_index.csv
   git commit -m "Record <EXPERIMENT> CodaBench result"
   git push origin <your-branch>
   ```
   `report/experiment_index.csv` has one row per run in `cs604-outputs`: config,
   status, commit, internal accuracy, public accuracy, submission ID,
   trainable parameters, and GPU hours.
4. **Write it up.** Add a dated entry at the top of `report/progress_log.md`
   saying what ran, the result, and the decision. Update `report/EXPERIMENTS.md`
   for runs that go in the report. Copy numbers from `run.json`,
   `metrics.csv`, `.inference.json`, and `leaderboard.json`; never from memory.

## What lives where

| Kind | Keep in | Notes |
| --- | --- | --- |
| Code, configs, notebook, index, logs | git (your branch) | never commit checkpoints, archives, or predictions |
| Finished runs, checkpoints, submissions | `cs604-outputs` on the laptop | the permanent archive; back it up |
| Inputs (`cs604/inputs`), tokens (`cs604/secrets`) | Drive | small; needed by every Colab run |
| Runs in progress | Drive `cs604/outputs` | temporary; sync, then prune |
| Course images | `cs604-course-data` on the laptop; `/content` on Colab | never on Drive |

## Checklist per experiment

- [ ] `configs/experiments/<EXPERIMENT>.yaml` committed and pushed
- [ ] Laptop screen run, or Colab run with `EXPERIMENT` set, finished with `status: completed`
- [ ] `sync_run.py <EXPERIMENT> --prune-drive` (Colab runs), then Drive trash emptied
- [ ] `predict.py` → `submissions/val_team_xx.zip`
- [ ] Uploaded; `record_submission.py` with the ID, and later the score
- [ ] `summarize_experiments.py`; index committed
- [ ] `progress_log.md` entry
