# Progress log

Newest session at the top. Each entry records what ran, which files went in and out, the result, and the decision. Training numbers still live in each output folder's `run.json`, `metrics.csv`, and `config.json`.

## 2026-10-04 — DINOv2 ViT-B/14 prototype comparison; laptop GPU power cap found

**Code (commit `2f96526`):** `image_size` is passed to timm (`img_size`) and the
transforms follow the built model (`model_data_config`), so DINOv2 runs at 224
instead of its 518 default. Configs `prototype_init_dinov2.yaml` and
`prototype_validate_dinov2.yaml`.

**Model check:** `vit_base_patch14_dinov2.lvd142m` is DINOv2's ViT-B: 85.72M
parameters at 224 (ViT-B/16: 85.80M), same depth/width/heads/MLP. At 224 it has
256 patches versus 196 for ViT-B/16 (about 30% more compute per image).
Pretraining: self-supervised on LVD-142M; its curation sources (DINOv2 paper,
Table 15) include ImageNet-22k, ImageNet-1k, Google Landmarks and fine-grained
sets such as CUB and Flowers, but no iNaturalist dataset. LVD-142M was retrieved
from about 1.2B uncurated web images, so web copies of iNaturalist photos
cannot be ruled out.

**Runs (laptop):** prototype build 53.758% top-1 on the fixed holdout (split
file byte-identical); evaluate-only gate 53.758% (`prototype_validate_dinov2`,
commit `2f96526`, `git_dirty` only from the notebook). Holdout loss at scale 20:
2.602 (augreg_in21k 4.227, orig_in21k 3.045).

| Backbone | Prototype top-1 |
| --- | --- |
| `orig_in21k` | 41.816% |
| `augreg2_in21k_ft_in1k` | 43.900% |
| `augreg_in21k` | 50.014% |
| `vit_base_patch14_dinov2.lvd142m` @224 | **53.758%** |

**Laptop GPU:** the old charger capped the RTX 5080 at 35 W (P5, memory clock
810 of 14,001 MHz; driver default limit 80 W, max 175 W). With the original
charger and Windows "Best performance" it runs at 174 W, P0, full memory clock:
prototype inference went from about 124 to about 1,000 img/s. Earlier laptop
throughput numbers (28–166 img/s, "~2 h" screens) were measured under the cap.

**Decision pending:** DINOv2 leads by 3.7 points but rule 6 (the Prof) asks to
start from the ImageNet-21k ViT-B/16; confirm before locking the base model.
This supersedes the 2026-10-03 "augreg_in21k is the base model" decision below.
Report write-up with all tables: `report/EXPERIMENTS.md`, "Base-model screen".

## 2026-10-03 — ImageNet-21k base model chosen: augreg_in21k

**Code and configs (commit `84b7902`):** `head_lr`, `train_crop_scale`, the
IN1k trained-head screen config, and prototype/evaluate-only configs for both
IN21k ViT-B/16 variants. `build_prototypes.py` stores its backbone and refuses
to overwrite; training rejects prototypes from another backbone.

**Runs (laptop, RTX 5080, commit `84b7902`; `git_dirty` only from the
uncommitted notebook):** 40-image prototypes per class, scored on the fixed
50,000-image holdout (split file byte-identical to `prototypes/splits/internal_val.txt`).

| Backbone | Prototype top-1 | Evaluate-only gate | Holdout loss at scale 20 | Build time |
| --- | --- | --- | --- | --- |
| `augreg2_in21k_ft_in1k` (previous) | 43.900% | 43.900% | — | 81 min |
| `orig_in21k` | 41.816% | 41.816% | 3.045 | 62 min |
| `augreg_in21k` | **50.014%** | **50.014%** | 4.227 | 48 min |

Outputs: `cs604-outputs\prototypes_<variant>\class_prototypes.{pt,json}` and
`cs604-outputs\prototype_validate_<variant>\` (`baseline.json`, `run.json`).

**Decision:** `vit_base_patch16_224.augreg_in21k` is the base model (rule 6),
with `expected_initial_accuracy: 0.50014`. Its prototypes alone are 1.08 points
below the 51.094% frozen-head LoRA screen. Its higher loss at the fixed scale 20
despite higher accuracy suggests it needs a larger cosine temperature, which the
trained head can learn. Next: the trained-head screen on `augreg_in21k`.

## 2026-10-03 — Full-budget frozen-head run, diagnosis, and new course rules

**Code (2026-10-02, commits `c5ac31d`, `8cb4732`):** `budget_48gb_hours` fits the
cosine schedule to a 48 GB-hour budget and stops training in time for the final
holdout evaluation; checkpoints carry elapsed time through a resume. The Colab
training cell flushes Drive before ending the runtime and arms a watchdog that
ends it if training hangs. See `WORKFLOW.md`.

**Run:** `configs/experiments/prototype_lora_r8_fp32_budget_40.yaml` on Colab,
A100 80 GB, commit `8cb4732`. Output:
`cs604-outputs\prototype_lora_r8_fp32_budget_40`.
Stopped by the budget after 83,308 updates (6.7 epochs), 1.901 normalized hours.
Internal top-1 **52.924%**, versus 51.826% for the 1-epoch run; accuracy levelled
off from epoch 4. The 80 GB card gave only 1.15 h of wall-clock budget and ran at
about 740 images/s.

**Diagnosis (evaluation-only, laptop):** `analysis/diagnose_best.json` in the run
folder. 61.55% top-1 on its own training images (underfitting); cross-entropy
lowest near cosine scale 40, not the frozen 20; mean cosine to the true prototype
0.58. The `baseline.pt` comparison was stopped at the 1-hour background limit
because the laptop evaluated at about 28 images/s.

**Decision from the Prof:** train the classifier head in every method; the head
is not counted in the 5% limit; any comparable-size ViT may be the backbone.
Recorded in `AGENTS.md` ("Course rules"). Earlier frozen-head runs become an
ablation. Next: `head_lr` support and a trained-head screen, then a prototype
screen of candidate backbones.

## 2026-10-01 — Colab data staging debugged; 40-image LoRA run started

Worktree: `cs604-prototype-baseline`, branch
`prototype-baseline`. Colab checks out the pinned commit
`78c28aa451e14525b79d666057f25793aa652c72`; the notebook cells come from
`4ed5686` (they run from the local notebook, not the checkout).

**Goal:** run `configs/experiments/prototype_lora_r8_fp32_40.yaml` on Colab Pro
(account `the team's Colab account`). The training archive is too large for the
Drive quota (assumed 15 GB), so the data must be staged on the Colab VM.

**Dataset facts (Hugging Face `doem1997/cs604-course-data`):**

| File | Size | Notes |
| --- | --- | --- |
| `train.tar.gz` | 40,115,732,248 bytes | images only; about 39 GB once unpacked, 450,000 JPEGs in 10,000 class folders |
| `train.txt` | 28,300,050 bytes | 450,000 lines; a **separate repo file, not inside the archive** |
| `val.tar.gz` | 4,443,572,618 bytes | not needed for training (holdout comes from `internal_val.txt`) |
| `val.txt` | 1,050,000 bytes | separate repo file |

**Storage plan:** Drive holds only the inputs (`cs604/inputs/`, about 32 MB),
secrets (`github_token.txt`, `hf_token.txt`), and run outputs. Course archives
are never stored on Drive. Each checkpoint stores the full model state, about
0.37 GB, so one 40-image run (about 12,500 steps, a checkpoint every 1,000)
needs about 6–7 GB of Drive. Delete the smoke-test output after it passes, and
empty the Drive trash, since trashed files still count against the quota.
The Colab VM disk is 236 GB.

### Failures and root causes

1. **`stage_data` exited 1 with no error message (every attempt, including the
   earlier Drive-cache setup).** Cursor's Colab kernel only shows what the notebook
   itself prints. Output from `subprocess.run` / `check_call` children is dropped,
   so neither `stage_data.py` nor `train.py` output was visible.
   Fix (`4ed5686`): the first notebook cell defines `run_logged()`, which relays
   child stdout and stderr line by line. The stage_data, smoke-test, and training
   cells use it. **Re-run the first cell after a kernel restart.**
2. **The data check could never pass on Colab.** `stage_data.py` required
   `train.txt` after extraction, but it is not in `train.tar.gz` and was never
   downloaded. This bug existed before this session's changes.
   Fix (`78c28aa`): `ensure_list_file()` downloads `train.txt` / `val.txt` from the
   Hub. Verified: the Hub copy is byte-identical to the desktop copy.
3. **VM disk filled up.** Failed attempts left 76 GB in `/content/cs604-course-data`,
   leaving 37 GB free, which is less than the 39 GB unpacked train split.
   Fix (`a103d2b`): in `--stream` mode, unverified extracts and the HF `.cache`
   folder are removed first, and staging refuses to start below 50 GiB free.
4. **Download-then-extract needs about twice the archive size** (about 80 GB peak).
   Fix (`cfb18c1`): `--stream` pipes `curl` from the Hub straight into `tar -xz`,
   so the `.tar.gz` never touches disk. The HF token is passed to curl on stdin,
   not in argv. Up to 3 attempts; each one restarts the full 40 GB download.

Earlier in the session Cursor added VM-only staging (`40d97d0`): no archives on
Drive, and the local archive is deleted after extraction.

### Reusing an earlier extract

When the stream fails after unpacking, the images are usually all there.
`images_match_list()` checks that every path in `train.txt` exists and is
non-empty. If so, stage_data writes the `.extracted_train.tar.gz` marker and
skips the download. This is how the run started: the VM already held all 450,000
images, so only `train.txt` was fetched. On the desktop dataset the check took
54 s.

### Commits (all pushed to `origin/prototype-baseline`)

| Commit | Change |
| --- | --- |
| `40d97d0`, `9ecceb8` | VM-only staging and its notebook pin (Cursor) |
| `cfb18c1`, `603f3ee` | `--stream` and pin |
| `a103d2b`, `be24379` | low-disk check, partial-extract cleanup, and pin |
| `78c28aa`, `a070d1e` | `train.txt` fetch, reuse of a complete extract, and pin |
| `4ed5686` | notebook relays script output |

Verification: `tests.test_pipeline` and `tests.test_stage_data` pass (13 tests).
A real-URL probe returned the gzip header and the 40.1 GB content length.
The Colab checkpoint smoke test passed (`last.pt` step 4, resume from step 2).

### Colab checklist

1. If `df -h /content` shows much less than 50 GB free, use
   **Runtime → Disconnect and delete runtime**, or remove `/content/cs604-course-data`.
2. Run the cells in order: the first cell (GPU check and `run_logged`), mount,
   token, checkout, inputs, stage_data (wait for `data ok … 450000`), smoke test,
   then training.
3. A runtime reset wipes `/content`. The next stage_data run downloads the archive
   again (about 20–40 min). Resume training from the latest verified
   `checkpoints/resume_step_*.pt` on Drive.

**Open question:** `du` reported 44 GB, and later 5.3 GB, under `/content/drive`,
which does not fit a 15 GB quota. Check the real quota of the Colab account at
drive.google.com; if it is larger, the Drive-only constraints above can be relaxed.

### Run result: `prototype_lora_r8_fp32_40.yaml`

Output: `/content/drive/MyDrive/cs604/outputs/prototype-lora-r8-fp32-40` (Drive).
Recorded commit `78c28aa451e14525b79d666057f25793aa652c72`, `git_dirty: false`.
Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GB VRAM).

- Rank-8 LoRA on `qkv`, frozen float32 cosine head initialized from the 40-image
  prototypes, one epoch over 40 training images per class, batch 32: **12,500
  optimizer steps** (= 400,000 / 32). Trainable parameters 294,912 (0.3145%).
- **Starting accuracy 43.93%** (about 21,965 / 50,000). This passed the gate
  (expected 43.90% ± 0.10 points). The desktop measured exactly 43.900%; the
  0.03-point gap (about 15 images) is attributed to bf16 numerics on a different GPU.
- **Best internal holdout: 51.826%** (25,913 / 50,000): **+7.90 points** over its
  starting score, and +0.73 points over the 10-image, two-epoch screen (51.094%).
- `elapsed_sec` **412.09** (6m 52s), covering the initial evaluation, training,
  the final evaluation, and every verified checkpoint copy to Drive. GPU-hours
  0.1145; normalized 48-GiB hours 0.2265 (about 11% of the 2 h × 48 GB budget,
  counting the full 95 GB card). Peak allocated memory 2.09 GiB.

The speed is consistent with the work done: about 500,000 images in 412 s is at
least 1,200 images/s overall. The desktop RTX 5080 Laptop runs reached only about
30 images/s training and 142 images/s evaluating. That is far below what that
GPU can compute, so the desktop runs were limited by data loading on Windows,
not by the GPU. The per-interval `images_per_sec` in this run's `metrics.csv`
should confirm the Colab rate.

## 2026-09-30 — rank-8 LoRA screen completed

Worktree: `cs604-prototype-baseline`, branch
`prototype-baseline`, recorded commit `910b2ff4d3cfcf89485243550fd429277d50aa95`
(`git_dirty: true`; the new config was not committed before the run).

**Run:** `configs/experiments/prototype_lora_r8_fp32_screen.yaml` through
`scripts/train.py` and the CUDA virtualenv. It was the only GPU job.

- Input: `cs604-outputs\prototypes\class_prototypes.pt`
  and the fixed 50,000-image internal holdout. The frozen cosine head stayed at
  those 40-image prototypes, with float32 class scores. LoRA rank 8 on attention
  `qkv` trained for two epochs on ten images per class (6,250 updates).
- Output: `cs604-outputs\prototype-lora-r8-fp32-screen`

**Starting accuracy:** **43.900%**, or **21,950 / 50,000**, loss 3.501497850341797.
`baseline.json` records `passed: true`, `accuracy_delta: 0.0`, and
`optimizer_steps: 0`. `baseline.pt` is that step-zero checkpoint.

**Final holdout:** **51.094%**, or **25,547 / 50,000**, loss 2.8203868717956544,
at step 6250. Epoch 1 was **50.508%** (25,254 / 50,000). The second epoch added
**0.586 percentage points**. Versus the 43.90% baseline this is **+7.194
percentage points** (3,597 additional correct images). `run.json` status is
`completed`, `best_internal_accuracy` is 0.51094, and `optimizer_steps` is 6250.
`best.pt` and `last.pt` both store step 6250. Logged elapsed time is
**7351.8314 s** (2h 2m 32s), including the initial evaluation. Peak allocated
memory is **2.1347 GiB**. GPU-hours 2.0421754; normalized 48-GiB hours 0.6773215.
Trainable parameters: 294,912 (0.3145%), all in the LoRA adapters.

Left unchanged: `prototypes/class_prototypes.pt`, `prototype-validate-fp32-head`,
and `prototype-head-screen`. No leaderboard file was written.

Decision: frozen-head LoRA rank 8 beats both the untouched prototypes and the
head-only screen, and it stays under 5% of parameters. Epoch 2's gain was much
smaller than epoch 1's. Do not rerun this config into the same output directory.

## 2026-09-30 — head-only screen completed

Worktree: `cs604-prototype-baseline`, branch
`prototype-baseline`, commit `929f8f361007a3ba90e08a21d6cd2b7592f12c08`
(`git_dirty: false`).

**Run:** `configs/experiments/prototype_head_screen.yaml` through
`scripts/train.py` and the CUDA virtualenv. When this session checked the GPU,
that command was already the only training process, so no second job was started.

- Input: `cs604-outputs\prototype-validate-fp32-head\best.pt`,
  the original prototypes, and the fixed 50,000-image internal holdout.
  The optimizer saw five training images per class for one epoch (782 updates).
  The backbone stayed frozen and class scores stayed in float32.
- Output: `cs604-outputs\prototype-head-screen`

**Starting accuracy:** **43.900%**, or **21,950 / 50,000**, loss 3.501497850265503.
`baseline.json` records `passed: true`, `accuracy_delta: 0.0`, and
`optimizer_steps: 0`. This matches the 43.90% prototype baseline before any
head update. `baseline.pt` is that step-zero checkpoint and was not rewritten
after 21:27:49.

**Final holdout:** **45.858%**, or **22,929 / 50,000**, loss 2.6540017303466796,
at step 782. That is **+1.958 percentage points** versus 43.90% (979 additional
correct images). `run.json` status is `completed`, `best_internal_accuracy` is
0.45858, and `optimizer_steps` is 782. Because the trained score is strictly
higher, `best.pt` and `last.pt` both store step 782. Logged elapsed time is
**1606.8395 s** (26m 47s), including the initial evaluation. Peak allocated
memory is **1.1328 GiB**. GPU-hours 0.4463443; normalized 48-GiB hours 0.1480375.

Left unchanged: `prototype-validate-fp32-head/best.pt` (SHA-256 still
`57214311e11c71b21bc79e92f629a8a01d23d5aa53c8187ec6be9d632159e1b2`), that run's
`baseline.pt`, the failed `prototype-validate` checkpoint, and
`prototypes/class_prototypes.pt`. No leaderboard file was written.

Decision: one epoch of head-only training improves the fixed holdout. It remains
a diagnostic screen. Initialization used 40 labeled images per class, and the
trainable head is 8.2158% of the model, above the under-5% adaptation target.
Do not rerun this config into the same output directory.

## 2026-09-30 — starting-point validation completed

Worktree: `cs604-prototype-baseline`, branch
`prototype-baseline`, base commit `010cf39aa28aef6987cdadaa15450e467b33f735`
with uncommitted implementation changes (`git_dirty: true`).

**First evaluation:** `configs/experiments/prototype_validate.yaml`, using the
existing prototypes and the exact original 50,000-image internal holdout.
Output: `cs604-outputs\prototype-validate`.
Top-1 **43.768%**, zero optimizer updates, 356.51 logged seconds. The check failed
because the score differed from 43.90% by 0.132 percentage points, outside the
predeclared 0.10-point tolerance. The diagnostic checkpoint and logs were preserved;
no `best.pt` was created.

**Correction and second evaluation:** `configs/experiments/prototype_validate_fp32_head.yaml`.
The original prototype script uses float32 normalization and similarity scoring,
whereas the cosine classifier previously scored inside bf16 autocast. Added an
explicit `cosine_head_fp32` option that keeps only the final scoring in float32.
The backbone remains bf16; model weights, split, prototypes, and tolerance are unchanged.

Output: `cs604-outputs\prototype-validate-fp32-head`.
Result: **43.900%**, or **21,950 / 50,000 correct**, with **zero optimizer updates**.
The accuracy gate passed. Logged run time: **351.93 seconds** (5m 52s), peak
allocated memory **1.0353 GiB**. Evaluation throughput: **142.62 images/second**.
This reproduces the original internal score; it is not a leaderboard result.

Saved `baseline.pt`, `best.pt`, `baseline.json`, `run.json`, `config.json`, and
`metrics.csv`. See `checkpoint_audit.json` for the post-run checkpoint checks.
Both attempts used the same prototype-file and holdout SHA-256 hashes.

Verification after the correction: **15 unit tests** and the **five-step synthetic
GPU smoke test** passed. The original failed evaluation is retained for diagnosis.

Decision: starting-point preservation and validation are complete. The prepared
head-only screen now loads `prototype-validate-fp32-head/best.pt` and retains
`cosine_head_fp32: true`. **The head-only training screen has not been started.**

## 2026-09-30 — prototype starting-point protection (implementation)

Worktree: `cs604-prototype-baseline`, branch
`prototype-baseline`. The original checkout is unchanged. See `HANDOFF.md` for
commands and cross-agent continuation instructions.

Added evaluation-only `configs/experiments/prototype_validate.yaml` and the separate
`prototype_head_screen.yaml`. The pipeline saves `baseline.pt` before updates,
checks the original holdout and starting accuracy, and includes step zero in
best-checkpoint selection. Fresh protected runs cannot overwrite existing results.

Verification: 13 unit tests passed using tiny CPU models and mocked evaluation
scores. The five-step synthetic GPU smoke test passed on the RTX 5080 Laptop GPU,
writing to `cs604-outputs\smoke-prototype-baseline`.
No real-data validation or training was launched, so **43.90% has not yet been
remeasured** by the new experiment.

## 2026-09-29 — pipeline, full fine-tune, prototypes

### What changed

Built the experiment pipeline in `cs604-group-project`: a fixed internal holdout, training and evaluation, LoRA / frozen-head / last-block modes, prototype initialization, prediction files, and one YAML per experiment under `configs/experiments/`.

### Runs

**Throughput check.** `scripts/train.py` with `configs/experiments/benchmark.yaml`.

- Input: `cs604-course-data`, 100 optimizer steps, 5 training images per class.
- Output: `cs604-outputs\benchmark\`
- Result: about 23 images/second after warm-up on the RTX 5080 Laptop GPU. No accuracy was expected from this short probe.

**Full fine-tune.** `scripts/train.py` with `configs/experiments/full_baseline.yaml`.

- Input: the same dataset. All 40 training images per class. The classifier started from random weights.
- Output: `cs604-outputs\full-baseline\` (`best.pt`, `last.pt`, `metrics.csv`, `run.json`)
- Result: internal top-1 **0.93%** after 12,500 steps and 4.45 hours. Peak allocated memory was 1.82 GB.

**Prototype initialization.** `scripts/build_prototypes.py` with `configs/experiments/prototype_init.yaml`.

- Input: the same dataset. Each class prototype is the average pretrained feature of its 40 training images. The 5 held-out images per class were not included in the average.
- Output: `cs604-outputs\prototypes\class_prototypes.pt` and `class_prototypes.json`
- Result: internal top-1 **43.90%** with no training, in 1.35 hours. The course zero-shot reference is 43.7%.

### Decision

A random 10,000-class head does not learn in one full fine-tune. Later training runs should start from `class_prototypes.pt`. The prepared config for training only that head is `configs/experiments/prototype_head.yaml`. No leaderboard file has been submitted.
