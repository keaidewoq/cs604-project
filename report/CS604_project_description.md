# CS604 Project Description

SMU Classification: Restricted

## Project overview

**Task:** Fine-grained multi-class classification, model adaptation (SFT, low-rank adaptation (LoRA), Adapter, AdaptFormer)

**Evaluation metrics:** Top-1 accuracy, parameter count, inference latency, training GPU-hours

**Constraints:** 2-hour training on 48 GB VRAM (for example, 2 × RTX 3090)

### Vision Transformer (ViT)

- Foundation transformer model in the computer-vision domain.
- Base model: IN21K → IN1K fine-tuned ViT-B/16
  - timm version: `timm.models.vit_base_patch16_224` (`pretrained=True`)  
    https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
  - Hugging Face version: `google/vit-base-patch16-224`  
    https://huggingface.co/google/vit-base-patch16-224

## Dataset

Anonymous dataset. Download manually from https://huggingface.co/datasets/doem1997/cs604-course-data

Multi-class image classification, 10,000 classes.

| Split | Samples | Per class |
| --- | ---: | ---: |
| Training | 450,000 | 45 |
| Validation | 50,000 | 5 |
| Test | 100,000 | Released in Phase 2. Submit predictions to CodaBench. |

**File structure**

- Images are in `train/`, `val/`, and `test/`.
- Labels are in `train.txt`, one `image_path class_id` per line.

## Runtime constraints (±10%)

- **Training GPU-hours:** 2 hours × 48 GB VRAM (for example, 2-hour training on 2 × 24 GB RTX 3090/4090).
- **Base model:** `vit-base-patch16-224-in21k`

## Tasks

### 1. Full fine-tuning

Constraint: 2 hours × 48 GB VRAM (or equivalent, for example 4 hours × 24 GB VRAM).

### 2. Efficient model adaptation

- Freeze the pre-trained model and train only a small set of parameters (for example, less than 5%) while improving performance.
- Reduced catastrophic forgetting, and expected to outperform full fine-tuning especially when the dataset is small and in few-shot situations.
- Possible directions:
  1. Selected-layer fine-tuning
  2. LoRA — adds low-rank matrices to frozen weights
  3. Adapter — inserts small bottleneck layers
  4. AdaptFormer — inserts small bottleneck layers into the FFN
  5. Prefix tuning — adds a trainable continuous prefix to the input
  6. BitFit — trains only the bias parameters
- Constraint: 2 hours × 48 GB VRAM (or equivalent, for example 4 hours × 24 GB VRAM).

## Evaluation metrics

- Top-1 classification average accuracy (calculated by CodaBench).
- Trainable parameters (for example, LoRA is 0.34% of the foundation model). Specify this in the report.
- Inference latency. Specify this in the report.
- Training time and VRAM. Provide a training log in the report.

The following performance was reported by a PhD student. It is neither the lower bound you are expected to meet, nor the upper bound. It is only a guide: results like 90%+ are not expected.

| Method | Top-1 average accuracy (%) | Trainable parameters (%) |
| --- | ---: | ---: |
| Zero-shot with an initialized classifier head | 43.7 | 0 |
| Full fine-tuning | 70.8 | 100 |
| Low-rank adaptation | 68.3 | 0.34 |

## Things to look into

- Optimization — bf16 precision, gradient accumulation
- Initialization — how to initialize the classifier head and PEFT adapters
- Loss design — which loss to use

## Instructions

Start from a decent open-source baseline and improve it. Submit results to:

https://www.codabench.org/competitions/18195/?secret_key=834843a4-0303-48db-b601-6520642bc531

**Result format**

- Name submissions `val_team_xx.txt` and `test_team_xx.txt`.
- One `image_path class_id` per line. Example: `test/image_000025.jpg 7919`

## Timeline

### Phase 1 — Validation

Submit the model's predictions on `val/` to CodaBench.

- 26 Sep 2026, 00:01: public leaderboard opens (maximum 5 submissions per day).
- 30 Oct 2026, 23:59: public leaderboard closes.

### Phase 2 — Test

Submit the model's predictions on `test/` to CodaBench.

- 31 Oct 2026, 00:01: private test data (`test/`) released.
- 06 Nov 2026, 23:59: private test submission due.
- 2 submissions in total.

### Presentation

- 07 Nov 2026, 12:00, during class.
- 15 minutes presentation + 5 minutes Q&A per group.
- Order will be set nearer the date.
- Slides should include figures, diagrams, and charts.
- Cover the investigation, the method's motivation, the design, comparison with baselines, and ablation studies.

### Final report

- 18 Nov 2026, 23:59.
- Submit on eLearn. Download the submission and check it.
- Format: CVPR conference paper LaTeX template  
  https://github.com/cvpr-org/author-kit/releases/tag/CVPR2026-v1

Marks are based on the following. Address each in the report.

- Motivation and justification of the design (important)
- What the design is (a framework diagram of the architecture helps)
- Ablation study on the parts of the design
- Results:
  - Top-1 average accuracy
  - Trainable parameter count
  - Inference latency
  - Training GPU-hours (implementation details)
  - Loss curves during training
