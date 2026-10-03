# Experiment direction after identifying iNaturalist 2021 mini

Date: 2026-10-02. Research and proposed experiments only; no new training run was launched.

## Update 2026-10-03: rules confirmed, order revised

The Prof confirmed three things (also in `AGENTS.md`, "Course rules"):

- The classifier head **must be trained** in every method, starting from the
  class prototypes.
- The head is **not counted** in the 5% limit; it applies to the backbone only
  (about 4.3M parameters for ViT-B).
- **Any vision transformer of comparable size** may be the backbone.
  Checkpoints trained on iNat2021 or TreeOfLife-10M stay excluded (leakage).

New evidence: the full-budget frozen-head run
(`cs604-outputs/prototype_lora_r8_fp32_budget_40`, 6.7 epochs, 1.90 normalized
hours) reached 52.924%, only +1.1 points over the 1-epoch run, and levelled off
from epoch 4. An evaluation-only check showed underfitting (61.6% top-1 on its
own training images), a frozen temperature that is too low (cross-entropy is
lowest near scale 40, not 20), and features far from the frozen prototypes
(mean cosine 0.58). Numbers and paths: `report/HANDOFF.md`.

Revised order (replaces the order below where they conflict):

0. **Base model: ImageNet-21k-only ViT-B/16** (Prof, 2026-10-03; `AGENTS.md`
   rule 6). Build prototypes for `vit_base_patch16_224.orig_in21k` (the PDF's
   `vit-base-patch16-224-in21k`) and `vit_base_patch16_224.augreg_in21k`, keep
   the better one, and run steps 1 and 3-6 on it. The existing
   `augreg2_in21k_ft_in1k` results become the IN1k-vs-IN21k comparison. A
   plausible reason IN21k-only features suit this task: they were trained on
   about 21,000 classes, including many plant and animal species, and were
   never narrowed to ImageNet-1k's 1,000 classes. Treat that as a hypothesis
   for the prototype comparison to test.

1. **Trained head on the current backbone.** Rank-8 `qkv` LoRA plus the
   prototype-initialized cosine head trained with its own learning rate; the
   temperature trains with the head. Screen at 10 images/class against
   `prototype_lora_r8_fp32_screen` (51.094%). Needs a `head_lr` option.
2. **Backbone screen without training.** Prototype accuracy on the fixed
   holdout for DINOv2 ViT-B/14, CLIP or SigLIP ViT-B/16, EVA-02 ViT-B,
   `vit_base_patch16_384`, and the current `augreg2` ViT-B/16 (43.90%). Needs
   prototype building and model loading for other timm ViT families. Record
   each model's pretraining data.
3. **Backbone capacity within 5%** on the best backbone: LoRA on all linear
   layers or higher rank, and AdaptFormer as the second course method.
4. **Crop policy and image detail** (sections 2–3 below), now with a trained
   head.
5. **Prototype-initialized full fine-tune** (section 1 below) on the chosen
   backbone, head trained, layer-wise learning-rate decay.
6. **Full-budget runs for finalists only**, preferably on a 40 GB A100
   (2.4 h of budget versus 1.2 h on an 80 GB card).

Every frozen-head result so far becomes the "frozen vs. trained head" ablation
in the report.

## Decision

*(Superseded 2026-10-03: the head is now trained; see the update above.)* Keep prototype-initialized ViT-B/16 with a frozen cosine head and rank-8 LoRA as the reference. Reframe the study around compute-limited species recognition across diverse organisms: classifier initialization, preservation of diagnostic image detail, and efficient feature adaptation. Prioritize a credible full-fine-tuning comparison and controlled crop experiments over a broad rank sweep.

## Dataset evidence and its implications

The official mini release has 500,000 training images, 50 for each of 10,000 species. Plants account for 4,271 species and insects for 2,526: together 67.97% of classes. Birds contribute 1,486; fungi and other animal groups make up the remainder. Thus species counts are balanced, while broad taxonomic groups are not. The original validation set has 100,000 images; its public test has 500,000. [Official dataset description](https://github.com/visipedia/inat_comp/tree/master/2021).

The course metadata instead specifies 450,000 labeled training images (45/class), 50,000 withheld validation images (5/class), and 100,000 withheld test images (10/class). Those counts are consistent with a repackaging of mini, as identified by the team, but do not independently establish the exact original-to-course split or class-ID mapping. Do not assume course class 123 is original species 123. Retain the course submission format and rules.

The dataset paper describes multiple life stages and appearances within a species, evidence such as nests rather than the organism itself, and images containing multiple species. Its augmentation discussion cautions about crops losing small subjects; the supporting bounding-box analysis concerns birds in iNat2017, not measured boxes for our course images. These findings motivate hypotheses, not a claim that a particular crop policy will improve our ViT. [Dataset paper, discussion and Appendix A](https://arxiv.org/html/2103.16483v2).

Consequences for our experiments:

- Preserve ordinary cross-entropy and uniform image sampling as the default. Class-frequency balancing has no obvious target when every species has the same number of training images. Broad-group reweighting would change the emphasis of the species-level objective.
- Test whether crops omit the organism or distinguishing parts. Preserve color cues initially; do not simultaneously introduce strong color changes, erasing, MixUp, and CutMix.
- Treat one centroid per species as a useful baseline, with possible limitations for different life stages or views. Test that limitation after the core comparisons.
- Taxonomy-aware losses and group metrics require a legitimate class-to-taxonomy mapping, which the local course metadata does not provide. They are conditional extensions, not the next implementation.
- Public source labels, image matching, filenames, or original archives must not be used to recover the withheld course labels. Dataset provenance informs model design; the existing evaluation boundary remains fixed. External biodiversity checkpoints and extra images are outside the current prescribed-backbone experiment. *(2026-10-03: other comparable-size ViT backbones are now allowed; checkpoints trained on iNat2021 or TreeOfLife-10M and extra iNaturalist images remain excluded.)*

## Current evidence, checked against local artifacts

- Original prototypes: **43.900%** internal top-1, built from 40 labeled images/class. This is supervised prototype construction, not zero-shot learning.
- Head-only screen: **45.858%**, with five optimizer-training images/class but the same 40-image prototypes. It is not a five-shot result.
- Rank-8 screen: **51.094%**, two epochs over ten optimizer-training images/class, frozen 40-image prototypes.
- New reference: **51.826%** (25,913/50,000) in `cs604-outputs/prototype_lora_r8_fp32_40/run.json` and `metrics.csv`. It began at **43.930%**, trained for 12,500 updates over 400,000 images, and has **294,912 trainable parameters / 93,773,569 total (0.3145%)**. Its audited prediction record points to `best.pt` with SHA-256 `e29422e7eb80df2b2a003b83a0d17a731981e57c35ce4200f1bc8b0782c8a37b`.
- Earlier random-head full fine-tune: **0.928%** in `cs604-outputs/full-baseline/run.json`. Different initialization, head, learning rate, and regularization confound comparison with LoRA. This demonstrates a poor configuration under its schedule, not that full fine-tuning is intrinsically inferior.

The 40-image run exceeds the ten-image screen by 0.732 percentage points, but also changes update count and schedule; this is not a controlled data-scaling result. The hyphenated output directory also reports 51.826%, with different code/timing metadata. Do not count these two directories as independent seeds without tracing their provenance. The prediction audit is not a public leaderboard score; none was established from the records inspected here.

## Ordered experiment plan

### 0. Lock the reference and the remaining budget

Use the same five-image-per-class internal holdout (split seed 604 and saved hash), the same prescribed pretrained checkpoint, and float32 cosine scores. Keep the existing 51.826% checkpoint immutable. Save per-image predictions for paired comparisons, top-1, top-5, loss, and confidence margins on the labeled internal holdout only. Inspect errors for missed subjects, confusing class pairs, and different within-class appearances without changing labels.

Audit cumulative compute before authorizing further runs: include prototype extraction, failed runs, screens, resumes, and any repeated executions according to the course accounting policy. The two 40-image directories report 347.70 and 412.09 logged seconds on a 94.97-GiB device; they do not establish the entire project's remaining budget. Keep setup, training, validation, checkpoint I/O, and inference timing distinguishable. Do not budget from peak allocated memory alone.

### 1. Required scientific comparison: full adaptation from prototypes

Start full fine-tuning and LoRA from the same original 40-image prototypes, with the same 224-pixel preprocessing, sample set, batch exposure, seed, and evaluation. Use a small full-backbone LR screen, initially 1e-5 and 5e-5, only if budget permits; these are proposed starting values, not established optima. Retain step-zero checkpoints and the accuracy gate.

Full fine-tuning trains the head and all backbone weights. Keep the initialization shared, while allowing method-appropriate LR. Report this as a method comparison, not a perfectly isolated parameter-count ablation. Give both methods the same compute allowance and report actual examples/updates reached; optionally include an equal-update comparison if affordable. Reserve budget for this before optional additions. The old random-head result remains a historical initialization baseline.

### 2. First dataset-driven ablation: crop policy

Use rank 8, frozen prototypes, 224 pixels, and the existing optimizer settings. Compare the actual resolved current crop policy against a gentler random-resized crop with area scale **(0.5, 1.0)**. Keep aspect-ratio range, flips, color jitter, erasing, sample subset, and schedule identical. Log the resolved transforms; do not infer them from timm defaults.

Screen on the same ten images/class and fixed 50,000-image holdout. Use the existing screen as a historical reference, but rerun a matched control if implementation or runtime changes prevent a clean comparison. Advance the better policy to all 40 training images/class only when its gain is credible. This tests subject retention; it does not establish that every organism should be tightly cropped.

### 3. Inference detail, then higher-resolution training

First compare center-crop inference with a two-view original-plus-horizontal-flip average at 224 on the saved reference. If error inspection suggests edge truncation, consider an aspect-preserving padded full-image view as a separate ablation. Report accuracy and latency; extra views are not free.

Next test **288 pixels versus 224** with the same crop policy and model weights, after implementing compatible model input sizing and positional-embedding handling. ViT-B/16 patch tokens increase from 196 to 324 (1.65x); the patch-token attention matrix grows about 2.73x, not necessarily the whole model's runtime. Benchmark actual throughput before training. An inference-only resolution test and a fine-tuned resolution test must be labeled separately. Consider 384 only if 288 helps enough to justify its cost.

### 4. One classifier experiment if errors justify it

Recompute prototypes from the adapted encoder using only the 40 training images/class; compare with its original frozen prototypes at identical inference settings. This tests whether centroids in the updated feature space help; they may also hurt because the encoder was optimized against the original head. Preserve both artifacts and count extraction cost.

If distinct appearance clusters remain problematic, test two train-derived centroids per species against one, using normalized features and a prespecified class-scoring rule. Never cluster with internal-holdout or official validation images. This is optional and lower priority than the full-tune and crop comparisons.

### 5. Capacity and loss changes only after the preceding evidence

Try rank 16 against rank 8 with alpha/rank held constant if training and validation evidence suggests insufficient capacity. A large rank grid is no longer the default priority. *(Superseded 2026-10-03: the head is not counted in the 5% limit and must be trained.)* A trainable dense 10,000 x 768 classifier alone exceeds 5% of this model, so do not add it to the PEFT method while claiming the current parameter budget. Count any learned head residual, scale, or auxiliary head explicitly.

Small label smoothing is a possible later single-variable test for overconfidence. Pure temperature scaling cannot change a single model's top-1 argmax; use it for calibration or ensemble weighting, not as an expected standalone accuracy gain. *(2026-10-03: during training the temperature does matter, because it shapes the gradients. The frozen value 20 was measured as too low; with a trained head it is learned.)*

## Selection and implementation gates

Use paired predictions and a class-resampled bootstrap for uncertainty on differences. Five holdout images/class make individual species rankings noisy; macro species accuracy equals overall accuracy here because the holdout is balanced. Use taxonomy aggregates only if an approved mapping becomes available. Repeat the finalist with another training seed if budget allows. An illustrative screening threshold is +0.3 percentage points, but uncertainty, reproducibility, and cost determine promotion; this is not a statistical guarantee. Keep the experiment count small to limit repeated holdout tuning.

Current code requires preparation before several proposed experiments:

- `src/cs604/engine/loop.py` forwards only `hflip`, `color_jitter`, `auto_augment`, and `re_prob` from YAML into train transforms. Crop scale needs an explicit supported setting.
- Real transforms use `resolve_model_data_config(model)`, and `build_vit` does not pass configurable input size. Merely changing `image_size` in YAML does not implement a real-data resolution ablation.
- Multi-view inference, adapted-prototype rebuilding, and multiple centroids require their own verified paths. Shared preprocessing must cover extraction, training, evaluation, and prediction.
- Create one YAML and fresh output directory per implemented experiment. Protect existing prototype/split hashes and step-zero gates. For a deliberately changed evaluation transform, record its separate baseline rather than forcing it through the old 43.90% expectation.

Proposed report question: **How much species-recognition accuracy can prototype initialization and parameter-efficient adaptation recover under a fixed compute budget, and how does preservation of image detail affect the result?** No accuracy target is promised from dataset identity alone.
