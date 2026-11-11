<!-- Snapshot of the shared Claude doc https://claude.ai/artifact/6Uj5g4m9U3XaWp1YsSuh9C at revision 502, taken 2026-11-11. Read-only copy for version history: edit the live doc, not this file. Images and embedded diagrams are not included; open suggestions are left out of the text and listed at the end. -->

# CS604 Final Report (Team 08)

2026-11-07 · @member

## Drafting notes

We draft content here first. Formatting comes last: the final version goes into the ICLR 2026 LaTeX template (`report/latex/main.tex` in the repo), due 21 Nov 2026.

- **Length:** 10 pages of main text. References do not count. The appendix goes after the references
- **Numbers:** every number must match the presentation slides. New material (loss curves, overfitting) is marked NEW.
- **Status:** every section now has a first draft. The Review to-do list at the end says what still needs checking.
- **How to help:** I think if you not sure, Claude can suggest and if you need me to review your suggestion, let me know. Anything you do not understand means the sentence in the report has to be reworked.
- **C****heckp****oint****: **Anastasia (read and reviewed up to end of section 4. I had Claude review and suggest changes to Section 5 to include more explanations as per Prof's requirements which needs to be approved)

## Abstract

*Draft, about 200 words.*

Pretrained Vision Transformers can be adapted to a new task by fine-tuning every weight or by training small added modules (parameter-efficient fine-tuning, PEFT). We compare the two on 10,000 fine-grained classes with 45 training images each, and a fixed budget of 2 hours on 48 GB. We ask how close PEFT gets to a well-tuned full fine-tune, how a fixed PEFT budget should be spent, and what limits the adapted model. A prototype-initialised cosine head, layer-wise learning-rate decay and training at 448 px take full fine-tuning to 84.07% top-1. AdaptFormer with LoRA, training 4.94% of the backbone, reaches 83.08%: within 1.0 percentage point, with less memory but no saving in training time and slower inference. Which parameters are trained matters more than how many. Most remaining errors confuse look-alike classes or show unfamiliar appearances, and the true class is usually in the top five. An attention-guided zoom and kNN re-ranking against individual training images add 2.0 points without retraining, and the best model loses only 0.2 points on unseen test images. Adapting a pretrained model is a joint design problem: resolution, optimisation, parameter allocation and prediction all interact.

## 1. Introduction

*S**econd** **draft, about **3**55** **words (under** 1 page). Scope: the motivation and the three research questions only; results are in** the abstract and Section 5**.*

Vision Transformers (ViTs) have achieved competitive or superior image classification performance compared with convolutional neural networks (CNNs), given sufficient pretraining data (Dosovitskiy et al., 2021). However, their limited inductive biases, such as locality, make training from scratch computationally expensive and data-intensive. Transfer learning addresses this limitation by adapting pretrained models to downstream tasks at substantially lower cost.

Two main adaptation strategies offer different trade-offs. **Full fine-tuning** updates all model parameters, providing maximum flexibility but requiring substantial training memory and risking the loss of useful pretrained representations (Kumar et al., 2022). **Parameter-efficient fine-tuning (PEFT)** instead updates a small subset of parameters, typically below 5%, while retaining most or all pretrained weights. This reduces trainable parameters and enables lightweight task-specific adaptations, but may constrain accuracy or introduce inference overhead depending on the method.

Choosing between these approaches requires more than comparing trainable parameter counts. Practical deployment is also constrained by training time, GPU memory and inference latency. Moreover, less is understood about how the placement and type of trainable parameters affect performance under a fixed resource budget, or whether residual classification errors can be addressed without further training.

We investigate these trade-offs on a challenging **10,000-class fine-grained image classification task**. Closely related classes exhibit high inter-class similarity, while variations in pose, life stage, lighting and background create substantial intra-class variation. With only 45 training images per class, adaptation is further constrained by limited labelled data. The difficulty of transferring pretrained representations is reflected in a nearest-class-centroid baseline, which achieves approximately 44% accuracy using pretrained features without adaptation. All experiments start from the same pretrained ViT-B/16 checkpoint (86M parameters), with each training run limited to **2 hours on a 48 GB GPU**.

This study addresses three research questions:

- **RQ1. Full fine-tuning versus PEFT:** Under the same computational budget, how do full fine-tuning and PEFT (under 5% trainable parameters) compare in accuracy, training efficiency, memory consumption and inference latency?
- **RQ2. Allocation of PEFT capacity:** How do the adaptation method, parameter placement and type of added capacity influence accuracy within a fixed parameter budget?
- **RQ3. Residual classification errors:** What errors persist after adaptation, and can training-free prediction-time techniques reduce them?

## 2. Related work

*Draft, about 1 page.*

**Pretrained Vision Transformers.** Vision Transformers (ViTs) achieve strong image classification performance through large-scale pretraining (Dosovitskiy et al., 2021). AugReg (Steiner et al., 2022) showed that strong augmentation and regularisation partly substitute for data; we start from its ViT-B/16 checkpoint, pretrained on ImageNet-21k and fine-tuned on ImageNet-1k. Full fine-tuning updates all model parameters, but when the classification head starts from random, early training can distort the pretrained features (Kumar et al., 2022). Kumar et al. propose LP-FT to reduce this: train a linear probe first, then fine-tune the whole model. An alternative is to start each class weight from the mean feature of that class, as in weight imprinting (Qi et al., 2018). This pairs naturally with a cosine classifier, which normalises features and class weights; CosFace adds a margin to make the learned features more discriminative (Wang et al., 2018). Layer-wise learning-rate decay gives earlier layers smaller updates to preserve their general features, and is standard in ViT fine-tuning recipes (Clark et al., 2020; Bao et al., 2022; He et al., 2022). FixRes (Touvron et al., 2019) shows that training and test resolution interact, and recovers accuracy by fine-tuning briefly at the test resolution. We adopt layer-wise learning-rate decay and a cosine classifier from this work, compare prototype initialisation against LP-FT, and test whether a fixed compute budget is better spent on resolution or epochs.

**Parameter-efficient fine-tuning.** PEFT methods differ in what they train. BitFit trains only the biases (Ben Zaken et al., 2022), while prefix tuning and VPT learn extra input tokens (Li & Liang, 2021; Jia et al., 2022). Both are tiny, but their capacity is limited, and extra tokens lengthen the sequence. Adapters insert small bottleneck layers (Houlsby et al., 2019), and AdaptFormer places one in parallel with the MLP (Chen et al., 2022). They add capacity, but stay in the forward pass and add latency. LoRA learns a low-rank update to the weights (Hu et al., 2022) that can be merged after training, so it adds no latency. DoRA adds a learned magnitude and comes closer to full fine-tuning, at a higher memory and time cost (Liu et al., 2024), and AdaLoRA moves rank between layers during training (Zhang et al., 2023). SSF, GPS and surgical fine-tuning instead choose which existing parameters to scale or train (Lian et al., 2022; Zhang et al., 2024; Lee et al., 2023).

**Prediction-time views.** Test-time augmentation averages predictions over flips or crops. It is cheap, but adds little when the extra views carry the same information. Fine-grained methods such as RA-CNN (Fu et al., 2017) and WS-DAN (Hu et al., 2019) use attention to find the discriminative region and look at it again at higher resolution. They recover small details lost at the input size, but usually train extra attention modules. Our zoom instead uses the trained model's own patch-to-class similarity to choose the region, so it trains nothing.

**kNN classification and re-ranking.** Nearest-neighbour classifiers compare a query with stored training features (Wu et al., 2018). Unlike class means or prototypes (Mensink et al., 2013; Snell et al., 2017), they keep a class's rare appearances. They need no training, but the feature store must be kept for each model. Like kNN-LM (Khandelwal et al., 2020), we add neighbour similarity to the model's own score.

**Positioning.** PEFT methods are usually compared at equal numbers of trainable parameters. We compare full fine-tuning and PEFT at an equal wall-clock budget, report memory, training time and latency separately, and treat the head's initialisation, the input resolution and the prediction step as part of the adaptation method.

*Reference details are in **`report/latex/refs.bib`**. Check each one against the paper.*

## 3. Method

*Draft, about 2.5 pages. Each design choice says what we did, why, and what we rejected.*

### 3.1 Setup and budget

We adapt ViT-B/16 (timm `vit_base_patch16_224.augreg2_in21k_ft_in1k`, 86M parameters) to 10,000 classes with 450,000 training images (45 per class). The classifier head always trains and is not counted in the 5% PEFT budget, which applies to the backbone (confirmed with the instructor).

Every final run trains on one A100 40 GB for 144 minutes and is stopped by the clock: 2.4 h × 40 GB = 96 GB-hours, the same as the brief's 2 h × 48 GB. The learning-rate schedule is fitted to that clock, so it reaches zero exactly at the time limit.

Three one-off steps run outside the 144-minute clock. None of them trains parameters, so we report them separately: class prototypes to initialise the cosine head (one forward pass of the frozen backbone over the training images), the AdaptFormer PCA initialisation for PEFT runs (8 batches, about 10 s), and the kNN feature bank of all 450,000 training images (0.7 GB, about 25–30 min at 448 px).

### 3.2 Overview

Under this budget we train two routes, full fine-tuning and PEFT (Figure 1). Both start from the same pretrained ViT, the same prototype cosine head and the same 448 px input, and both produce a model that goes through the same prediction steps. This lets us compare full fine-tuning and PEFT under identical conditions (RQ1). Our changes fall into four groups: how the head starts, how fast each block learns, how much detail the model sees, and how predictions are made.

**Figure 1 (draft).** Overview of the method. Full fine-tuning and PEFT start from the same pretrained ViT, prototype cosine head and 448 px input, and the trained model from either route goes through the same prediction steps (zoom and kNN re-ranking).

*[Embedded diagram not exported: Figure 1 draft · training routes and prediction pipeline]*

### 3.3 A full fine-tuning recipe that keeps the pretrained features

Our initial full fine-tune, which used a single small learning rate for all blocks and no schedule, reached 61.3% on CodaBench, below a linear probe on frozen features (62.5%). Following Kumar et al. (2022), we attribute this gap to feature distortion: the fine-tune degraded the pretrained features faster than it adapted them. A fair comparison with PEFT (RQ1) requires a full fine-tuning baseline that avoids this, so we made four changes.

- **Prototype cosine head.** Each class weight starts at the normalised mean pretrained feature of that class's training images. Logits are cosine similarities times a learned temperature. *Why not LP-FT:* a trained probe is confident, so it passes small gradients to the backbone; prototypes are reasonable but not confident, so the backbone keeps adapting (+1.6 pp over LP-FT, Section 5.2).
- **Layer-wise learning-rate decay.** Each block learns at 0.75× the rate of the block above: 4e-4 at the top, about 1.3e-5 at block 0. Standard in ViT fine-tuning (Bao et al., 2022; He et al., 2022), it protects the general features in early blocks, which large updates from 45 images per class could overwrite. The factor follows the MAE recipe; the top rate came from a five-point sweep (Appendix A).
- **Head learning rate 10× the backbone (4e-3).** The head is new and must move faster than a backbone that only needs nudging (+1.5 pp, Section 5.2).
- **Schedule fitted to the clock.** A 5% warm-up, then cosine decay positioned by elapsed time, so the learning rate reaches zero exactly at the 144-minute limit. Otherwise a time-limited run stops mid-decay at a high learning rate.

Settings we did not tune follow the MAE fine-tuning recipe for ViT-B (He et al., 2022): label smoothing 0.1 (Szegedy et al., 2016), drop path 0.1 (Huang et al., 2016) and AdamW weight decay 0.05 (Loshchilov & Hutter, 2019), with no decay on norms and biases (He et al., 2019). We clip gradients at norm 1.0, as in the original ViT fine-tuning (Dosovitskiy et al., 2021). Batch size 256 and bf16 were chosen to fit the 48 GB budget. Together these changes take full fine-tuning at 224 px from 66.3% to 75.7% on CodaBench (Section 5.2).

### 3.4 Trade-off between epochs and resolution

With 16×16 patches, a 224 px input gives 196 patches and a 448 px input gives 784. A small subject that covers 12 patches at 224 px covers about 35 at 448 px. Small subjects are a common cause of error (Section 5.5), and 79% of the images have a short side of 300–400 px, so training at 224 px throws away detail that is in the file. The cost is compute: with four times as many patches each step is slower, so fewer epochs fit in the budget.

Under a fixed budget, resolution and epochs therefore trade off. We train at 448 px: extra epochs gave diminishing returns, while at equal compute accuracy kept rising with resolution up to 448 px and flattened after it. Training most epochs at low resolution and switching to high resolution later did worse than training at the high resolution throughout (Section 5.2).

### 3.5 PEFT design: AdaptFormer + LoRA under 5%

The PEFT model freezes the backbone and adds three things, 4,262,412 parameters in total, or 4.94% of the 86,250,240-parameter backbone:

- **AdaptFormer** (Chen et al., 2022; 3,550,476 parameters): a bottleneck branch beside each block's MLP (768 → 192 → 768) with a ReLU and a learned per-block scale. The up-projection starts at zero, so training starts exactly at the pretrained model. The down-projection is initialised by PCA of the features.
- **Trainable biases and LayerNorm weights** in every block (122,112 parameters, 0.14% of the backbone).
- **LoRA rank 16** on each block's qkv projection (589,824 parameters).

The percentage is new trainable backbone parameters divided by the original backbone; the 10,000-class cosine head (7,680,001 parameters) is excluded, as the brief allows. A new task therefore needs about 17 MB of adapters plus a 31 MB head, instead of a 375 MB model.

The head and training recipe are those of Section 3.3, plus a CosFace margin of 0.1 on the cosine head.

*Why this design.* We added one change at a time on the holdout and kept only those that helped (Section 5.3). Two findings shaped it: once every block has some cheap capacity, where the parameters go barely matters, and the last 0.7% of the budget helped when spent on LoRA but not when spent on wider adapters.

LoRA alone at the same budget tied with AdaptFormer + LoRA. We kept AdaptFormer because it trains about 20% faster; the cost is slower inference, since its branch cannot be merged into the weights.

### 3.6 Prediction-time extensions (no training)

**Zoom: a second, closer view.** After the first pass, we build a class activation map from the cosine similarity between each patch token and the predicted class's head weight. We take the smallest window holding 70% of the map, crop it from a 2× upscaled copy of the image, and classify it again with the same model. The two softmax outputs are averaged. *Why:* tiny subjects and subjects outside the centre crop are common causes of error (Section 5.5).

**kNN re-ranking.** The model proposes its top K = 5 classes. Each candidate class c is re-scored by how closely the query matches that class's individual training images (Figure 2):

```latex
s(c) = \log p(c \mid x) + \beta \cdot \frac{1}{m} \sum_{i \in \text{top-}m} \cos\big(f(x), f(x_i^{c})\big), \qquad K = 5,\; m = 3,\; \beta = 20
```

Here f is the trained model's feature and x_i^c are the training images of class c. The feature bank holds all 450,000 training images (0.7 GB in fp16) and is built once. *Why:* in 64% of errors the true class is still in the top five (Section 5.5). Individual training images keep a class's rarer appearances, which a single head weight per class averages away. K, m and β were tuned on the labelled holdout (Section 4).

[image not exported: How kNN re-ranking scores a photo]

*Figure 2: How kNN re-ranking scores a photo. The model's **top K = 5 classes are re-scored by adding β = 20 times the mean of the m = 3 highest* *cosine similarities between the photo's feature and that class's training images.*

## 4. Experimental setup

*Draft, about 0.5 page.*

**Evaluation protocol.** We screen ideas cheaply, confirm them at full scale, and submit only selected systems. CodaBench allows few submissions (two in Phase 2), and a leaderboard score mixes many changes at once, so we answer causal questions with controlled runs on our own labelled data and use CodaBench only for leaderboard questions.

| Stage | Data | Used for |
| --- | --- | --- |
| Pilot | 1,000 classes, 3–5 epochs | Fast screening of many configurations |
| Labelled holdout | All 10,000 classes, 5 training images per class held out (50,000 images) | Tuning and controlled comparisons |
| CodaBench Phase 1 | 50,000 validation images, 5 per class | Comparing the final systems (RQ1) |
| CodaBench Phase 2 | 100,000 test images, 10 per class | Checking the best model for overfitting (Section 5.4) |

Pilot accuracies (about 90%) are inflated because there are ten times fewer classes to confuse, so pilot results are only compared with other pilot results. Holdout runs that select a setting train on the remaining 40 images per class; the final runs train on all 45. Differences under 0.4 pp on the holdout are treated as ties, since run-to-run noise is about 0.3 pp.

**Metrics.** Top-1 accuracy (CodaBench), trainable share of the backbone, peak training memory, training GPU-hours and inference latency. Latency is measured on an A100 in bf16 as the forward pass on inputs already on the GPU, with 30 warm-up and 200 timed iterations, excluding image loading and preprocessing. Batch-1 latency moves by about 1 ms between sessions, so the two final systems were timed in the same session.

**Implementation.** PyTorch and timm, AdamW, bf16 autocast. No CodaBench labels were used: Phase 1 serves as the validation set and Phase 2 as the test set, and both are scored only on the server.

*Open: total tuning compute outside the finals (laptop screens, A100 pilots), per person.*

## 5. Results and analysis

*Draft**, about 4 pages**.** *

### 5.1 Main results

Full fine-tuning beats PEFT both without extensions (82.5% vs 81.2%) and with zoom and kNN (84.07% vs 83.08%, Table 2). PEFT trains 4.94% of the backbone and needs 3.1 GB less peak memory, but it completes the same 5.5 epochs in 144 minutes, so it saves no training time: gradients still flow through the whole frozen backbone. It is also slower at inference. LoRA merges exactly into the weights, but the AdaptFormer branch cannot be merged, so PEFT takes 11.27 ms per image against 8.47 ms (13.45 ms before merging LoRA).

*Table 2: final systems on CodaBench Phase 1, 448 px, 144 training minutes on one A100. Latency on one A100 session, batch 1.*

| Metric | Full FT | AF + LoRA (PEFT) |
| --- | --- | --- |
| Phase 1 top-1, no extensions / + zoom / + zoom + kNN (%) | 82.5 / 83.22 / 84.07 | 81.2 / 82.2 / 83.08 |
| Trainable share of backbone | 100% | 4.94% |
| Peak training VRAM (GB) | 17.80 | 14.66 |
| Epochs in 144 min | 5.5 | 5.5 |
| Latency, one pass (ms, A100, batch 1) | 8.47 | 11.27 (13.45 before merging LoRA) |
| Latency with zoom (ms) | 23.43 | 29.88 |

**Training loss curves** (required by the brief). Epoch-mean training loss of the two final runs:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 (partial) |
| --- | --- | --- | --- | --- | --- | --- |
| Full FT, 448 px | 3.76 | 2.95 | 2.57 | 2.24 | 2.02 | 1.95 |
| AF + LoRA, 448 px | 5.67 | 5.20 | 5.05 | 4.83 | 4.55 | 4.35 |

Both curves fall smoothly and are still falling at the 144-minute limit, so training was stable and both models were still learning when the budget ran out. The two levels are not comparable: the PEFT loss includes the CosFace margin times a learned temperature, which grows from 20 to about 68 during training. The finals train on all 45 images per class, so they have no validation curve. Whether more epochs would have been a better use of the budget is answered elsewhere: at 224 px, going from 6 to 12 epochs added only +0.53 pp on the holdout, and higher resolution beat extra epochs at equal compute (Section 5.2). *Source: Van's draft, Section 5.5. Per-step plots: **`CS604_RUNS/<run>/figures/<run>_loss.png`** on Drive.*

### 5.2 Full fine-tuning ablations

The recipe and the resolution account for almost all of Task 1's score. From the 66.3% baseline at 224 px, the recipe of Section 3.3 adds 9.4 pp (to 75.7%), moving to 448 px adds 6.8 pp (to 82.5%), and zoom and kNN add 0.72 and 0.85 pp (to 84.07%). The resolution step went 75.7% → 80.7% (336 px) → 81.7% (384 px) → 82.5% (448 px); its runs are not perfectly matched.

*Table 3: full fine-tuning ablations on the 10,000-class holdout, 224 px, 3 epochs unless stated.*

| Setting | Holdout top-1 (%) | Change |
| --- | --- | --- |
| Pretrained features, prototype head, no training | 43.90 |  |
| Control: head at the backbone learning rate | 69.83 |  |
| + head learning rate 10× | 71.37 | +1.53 |
| No layer decay, same top LR (vs control) | 63.12 | −6.71 |
| Prototype head instead of LP-FT | 72.98 | +1.61 |
| 6 / 12 epochs instead of 3 | 73.97 / 74.49 | +0.98 / +0.53 more |
| 336 px instead of 224 px | 78.39 | +5.4 |

**Resolution at equal compute.** With 10 training images per class and the same compute for every run, holdout accuracy rose from 63.20% at 336 px to 64.06% (384 px), 64.79% (448 px) and 64.98% (512 px). The gain flattens at 512 px (+0.19 pp), so 448 px is our trade-off. Saving compute with low resolution lost in every test: progressive 224 → 384 px training was 4.3 pp below plain 384 px despite 2.5× the epochs, and a 224 px run with a short 336 px finish was 1.85 pp below plain 336 px. Higher resolution helps only when the backbone adapts: frozen features get worse as resolution rises (prototype accuracy 43.9% at 224 px, 37.2% at 448 px).

[image not exported: Full fine-tuning recipe changes, one at a time]

*Figure 4: Full fine-tuning recipe changes, one at a time, on Van's holdout (top-1 %, one seed, layer decay 0.75 throughout). Crop scale, colour jitter and a 30× head learning rate were ties.*

[image not exported: Resolution at equal compute]

*Figure 5: Resolution at equal compute (10 training images per class, the same GPU budget per run, so higher resolution fits fewer epochs). Full fine-tuning flattens after 448 px; AdaptFormer keeps rising up to 448 px (512 px not run).*

### 5.3 PEFT comparisons

**Method comparison (RQ2).** At 336 px and the same 144-minute budget, six methods tie between 77.6% and 78.5%, while methods that train less of the network trail (Table 4). *Which* parameters are trained matters more than *how many*: BitFit trains 0.12% of the backbone and reaches 73.4%, while last-block attention trains 2.75% and reaches 67.9%. Methods that touch every block beat those concentrated in one place. On a 1,000-class pilot, AdaLoRA was 2.7 pp below LoRA, so it was not run at full scale.

*Table 4: PEFT methods on CodaBench Phase 1, 336 px, plain view, 144-minute budget. Recipes and batch sizes differ, so close scores are ties. Latency: forward pass, batch 1; merged values in brackets.*

| Method | Backbone trained | Top-1 (%) | Peak VRAM (GiB) | Latency (ms) |
| --- | --- | --- | --- | --- |
| Last-block attention | 2.75% | 67.93 | 1.43 | 7.76 |
| Prefix, length 50 | 1.07% | 72.1 | 7.97 | 9.64 |
| BitFit, all biases | 0.12% | 73.4 | 7.01 | 7.40 |
| Serial adapter w96 + extras | 4.68% | 77.6 | 15.54 | 11.60 |
| GPS | 4.63% | 77.9 |  |  |
| AdaptFormer w192 | 4.13% | 77.9 | 14.53 | 10.28 |
| DoRA r24 + extras | 4.75% | 78.3 | 28.06 | 23–24 (7.71) |
| LoRA r24 | 4.12% | 78.4 | 19.30 | 15.37 (7.78) |
| LoRA r24 + extras | 4.65% | 78.5 | 19.40 | 14.2–15.0 (7.68–8.35) |

Trainable share predicts neither memory nor latency. BitFit trains 0.12% but still needs 7.0 GiB, because gradients flow back through every block; last-block attention stops the gradient at block 12 and needs only 1.43 GiB. Adapters, AdaptFormer and prefixes stay in the forward pass, while LoRA and DoRA cost nothing once merged.

**Where the budget goes.** Training biases and LayerNorms in every block added 0.50 pp to AdaptFormer (76.26% → 76.76%) for 0.14% of the backbone. At equal budget, spreading parameters over all 12 blocks or the last 8 tied, for LoRA (77.26% vs 77.33%) and for AdaptFormer (76.77% vs 76.90%), over two seeds.

**What kind of capacity.** Starting from AdaptFormer at 4.27% (76.77%), widening the adapters to 4.96% gave no gain (76.84%), while adding LoRA to the same budget gained 0.6 pp in both seeds (77.43%). LoRA alone at that budget tied with AdaptFormer + LoRA (77.26%).

**Final PEFT design.** Step by step on the holdout (336 px, 3 epochs): 75.27% for AdaptFormer with the default recipe, +0.99 pp with the cosine margin and design changes, +0.50 pp with biases and LayerNorms, +0.66 pp with LoRA, giving 77.43% at 4.94%. On CodaBench the same path went 77.9% (336 px) → 79.4% (+ biases and norms, 384 px) → 80.4% (448 px) → 81.2% (+ LoRA, 448 px), and 82.2% with zoom.

### 5.4 Overfitting: Phase 1 vs Phase 2 (NEW)

We check for overfitting by comparing the best model's Phase 1 and Phase 2 scores. Phase 1 guided every design choice; Phase 2 is 100,000 unseen images.

| Best model: full FT, 448 px | Phase 1 (50,000 images) | Phase 2 (100,000 images) | Change |
| --- | --- | --- | --- |
| Zoom | 83.22 | 82.91 | −0.31 |
| Zoom + kNN (final) | 84.07 | 83.84 | −0.23 |
| kNN gain | +0.85 | +0.93 |  |

Accuracy drops by only 0.2–0.3 pp, and the kNN gain holds (+0.93 vs +0.85). The model shows little sign of overfitting to Phase 1.

**Fit vs generalisation.** AdaptFormer fits its own training images almost as well as full fine-tuning (95.0% vs 95.5% top-1 on images it trained on) but scores 1.4 pp lower on held-out images. So the PEFT gap comes from how well the features generalise, not from too little capacity to fit the data. This matches wider adapters giving no gain (Section 5.3). *[Van to confirm: which runs (resolution, epochs, holdout) and the holdout accuracies behind the 1.4 pp. The source is Van's draft, Section 2.5; the figure is not in the repo logs. With the holdout numbers we can also report each model's train-holdout gap.]*

### 5.5 Error analysis

About half of the remaining errors confuse look-alike relatives, and about half show an appearance unlike any training image of the class (RQ3). We analysed the 11,111 errors of LoRA r24 on the all-class holdout (top-1 77.8%).

**Four kinds of error.** Figure 6 shows one example of each: look-alike relatives (two dowitcher species), a tiny subject (a hoverfly labelled as the flower it sits on), a subject outside the centre crop (a spider), and an unfamiliar look (a coneflower in a state unlike its training images). These are hand-picked illustrations, not frequencies.

[image not exported: Real mistakes, one of each kind]

*Figure 6: One real mistake of each kind: the holdout photo, training images of the predicted class and training images of the true class. Hand-picked illustrations, not frequencies.*

**How often each cause occurs.** We flag each error using the model's own features. An error can carry several flags, so the shares add to more than 100%.

| Flag | Meaning | Share of errors | Error rate among images with the flag |
| --- | --- | --- | --- |
| Sibling | The wrong prediction is a closely related class | 50% (5,539) |  |
| Unseen look | The image is unlike every training image of its class | 52% (5,763) | 60% |
| Two looks | The image shows the class's rarer appearance | 5% (538) | 38% |
| Messy class | The class's training images overlap another class | 77% (8,537) | 29% |
| No flag |  | 4% (436) |  |

Sibling and messy class reflect inter-class similarity; unseen look and two looks reflect intra-class variation. Frequency is not risk: messy-class images are 58% of the holdout and fail 29% of the time, close to the 22% average, while unseen-look images are 19% of the holdout and fail 60% of the time.

**The right answer is usually close.** Top-1 accuracy is 77.8%, but top-2 is 86.0%, top-3 89.2%, top-5 92.0% and top-10 94.4%. In 37% of errors the true class is ranked second, and in 64% it is in the top five. Re-ordering the short list could therefore fix many errors without retraining, which motivates kNN re-ranking.

### 5.6 Prediction-time extensions

Zoom and kNN re-ranking each help without retraining, and their gains add up: +2.0 pp together on the holdout (78.31% → 80.27%).

**Choice of second view.** On the holdout (full fine-tuning at 336 px, 3 epochs), a single view scores 78.80%. Adding a flipped view gives 78.97% (+0.17), a centred box of the same size as the zoom box gives 79.43% (+0.63), and the aimed zoom gives 79.59% (+0.79). The centred box isolates the effect of zooming in; aiming the box at the evidence adds the rest. At the same cost of two passes, the aimed zoom gives about four times the gain of a flip. Zoom can also break a correct answer when the deciding cue is outside the box.

[image not exported: Real zoom boxes that changed the answer]

*Figure 9: Real zoom boxes on the all-class holdout (LoRA at 336 px) that changed the model's answer.*

**Choice of kNN variant.** Comparing the query with individual training images beats comparing it with a summary of the class. The weight β was chosen on the other half of the classes.

| Variant | Compares the query with | Gain (pp) | Gain on top of zoom (pp) |
| --- | --- | --- | --- |
| kNN, mean of 3 (chosen) | Mean cosine to the class's 3 closest images | +1.32 | +1.23 |
| kNN, best match | Cosine to the single closest image | +1.18 | +1.02 |
| Weighted vote | 50 nearest images of any class | +1.14 | +1.03 |
| 3 k-means centres | Best cosine to 3 centres of the class | +0.70 | +0.69 |
| Class mean | Cosine to the class's average feature | +0.55 | +0.49 |
| kNN, mean of all | Mean cosine to all of the class's images | +0.13 | +0.13 |

Averaging a class erases its rare appearances; individual neighbours keep them.

**Together, and on CodaBench.** On the holdout (LoRA r24), zoom adds 0.7 pp, kNN 1.3 pp and both 2.0 pp, so the two extensions repair different errors. On CodaBench Phase 1 the kNN gain held for every model, with the same settings: LoRA at 336 px 78.42% → 79.82% (+1.40), full fine-tuning with zoom 83.22% → 84.07% (+0.85), and PEFT with zoom 82.20% → 83.08% (+0.88).

**Cost.** kNN adds under 0.1 ms per image once the bank exists, but the 0.7 GB bank takes 25–30 minutes to build at 448 px, must be rebuilt whenever the model changes, and can only fix errors whose true class is in the top five. On the holdout it breaks about one correct answer for every two it fixes. Zoom costs 2.5–2.8× a single pass at batch 1 (23.43 vs 8.47 ms for full fine-tuning).

[image not exported: Phase 1 and Phase 2 results and kNN trade-offs]

*Figure 12: Full fine-tuning at 448 px on CodaBench Phase 1 and Phase 2, with what kNN gives and what it costs. Items marked * were measured on the LoRA all-class holdout, not this model.*

## 6. Discussion and conclusion

*Draft, about 0.75 page.*

**RQ1: full fine-tuning versus PEFT.** A well-tuned full fine-tune beats the best PEFT model by about 1.0 pp (84.07% vs 83.08%) and is faster at inference. PEFT trains 4.94% of the backbone and needs less memory, but it saved no training time: both completed 5.5 epochs in the same 144 minutes. The course brief expected PEFT to win on small data. We think our task is not small in the sense that matters: 45 images per class across 10,000 classes is 450,000 images. On a 1,000-class pilot LoRA matched or beat full fine-tuning, while at full scale full fine-tuning won by 2.2 pp at 336 px, which points to total dataset size rather than images per class.

**RQ2: spending a PEFT budget.** Which parameters are trained matters more than how many. Cheap biases and LayerNorms in every block are worth keeping; once every block can adapt, placement barely matters; and LoRA is a more efficient way to add capacity than wider adapters. The head setup and learning rates mattered more than the choice between the strongest PEFT methods, which tied within 1 pp.

**RQ3: what limits the adapted model.** The remaining errors are mostly look-alike relatives and unfamiliar appearances, and the true class is usually in the top five. Changing the loss to push relatives apart did not help in our pilots. What helped added information: more pixels in training, a zoomed second view, and comparison with individual training images. These steps train nothing and their gains held on unseen test images.

**Directions ruled out.** More low-resolution epochs (448 px at fewer epochs beat lower resolutions at more); concentrating LoRA in later blocks (a tie at equal budget); parameter count as a proxy for quality (BitFit at 0.12% beat last-block fine-tuning at 2.75% by 5.5 pp); flip as the second view (+0.2 vs +0.8 pp for the aimed zoom); and small PEFT tweaks before fixing the head.

**Limitations.** CodaBench runs use one seed. The resolution step of Task 1 compares runs that are not perfectly matched. In the PEFT method comparison, recipes and batch sizes differ, so scores within about 1 pp are ties. Error flags are computed from the model's own features. The kNN bank is tied to one model and must be rebuilt when it changes, and its 25–30 minute build sits outside the training budget.

**Conclusion.** Adapting a pretrained ViT is a joint design problem: resolution, optimisation, where the trainable parameters sit and how predictions are made all interact. We recommend evaluating training compute and trainable parameters separately, treating the classifier head as part of the adaptation method, and testing robustness on unfamiliar appearances, which fail almost three times as often as the average image (60% vs 22%).

## Appendix and open items

Appendix (after the references): A. LR sweep · B. Full FT recipe table · C. Head × layer decay · D. Per-method PEFT tuning grids · E. Error-flag definitions and error rate by flag · F. Training logs (time, VRAM, GPU-hours) · G. Team contributions.

### Appendix figures (moved from the main text to save space)

[image not exported: Full fine-tuning vs AdaptFormer + LoRA on each measure]

*Figure 3: Full fine-tuning vs AdaptFormer + LoRA on each measure (accuracy: CodaBench Phase 1 with zoom + kNN; latency: A100, batch 1, LoRA merged). Each row has its own scale. PEFT wins on trainable share and training memory, full fine-tuning on accuracy and latency, and both fit 5.5 epochs in 144 minutes.*

[image not exported: Error flags as a share of all holdout photos]

*Figure 7: Each error flag as a share of all 50,000 holdout photos.*

[image not exported: Top-k accuracy on the holdout]

*Figure 8: Top-k accuracy on the holdout: top-1 77.8%, top-5 92.0%, top-10 94.4%. The right answer is usually on the short list.*

[image not exported: Zoom and kNN on the all-class holdout]

*Figure 10: Zoom and kNN on the all-class holdout: 78.3% with a single view, 79.0% with zoom, 79.6% with kNN and 80.3% with both. The two gains nearly add up.*

[image not exported: The kNN variants compared]

*Figure 11: The kNN re-ranking variants compared (****details in the table above**).*

Open items:

- [ ] Pull `step_log.csv` / `epoch_log.csv` for both final runs from `CS604_RUNS/` on Drive and plot the loss curves
- [ ] Get per-epoch training vs holdout accuracy for one full FT and one PEFT run (overfitting analysis)
- [x] Phase 2 numbers for the PEFT system (not needed: both Phase 2 submissions went to full FT)
- [ ] Final version of Figure 1 (pipeline) for the paper
- [ ] Confirm with the course that the appendix does not count towards the 10 pages
- [ ] Check every reference against the paper
- [ ] Van: share the run record for "one epoch with a decaying schedule beat 11 epochs at a constant learning rate" (Van's laptop holdout / EXPERIMENTS.md). Until then, the Method table only states why the schedule is used. Note: the CodaBench jump 0.621 → 0.663 cannot be credited to the schedule alone (progress log, 5 Oct).
- [ ] Van: share the runs behind "AdaptFormer 95.0% vs full FT 95.5% on training images, 1.4 pp worse on holdout" (Section 5.4): resolution, epochs, holdout and both holdout accuracies.

## Review to-do (for the team)

Tick an item when it is done. Van's two items are in the Open items list above.

**Numbers to check**

- [ ] PEFT build path on the holdout: the doc uses the slides' two-seed means (77.43, +0.66); Van's draft has single runs (77.33, +0.57). Pick one and use it everywhere.
- [ ] Slides 9 and 29 label the 7.7-epoch point as 224 px, but Van's resolution table puts 7.7 epochs at 336 px (336 / 384 / 448 / 512 px = 7.7 / 5.5 / 4.1 / 2.9 epochs). Check which is right before quoting "448 px at 4.1 epochs beat 224 px at 7.7".
- [ ] ~44% in the Introduction: nearest-class-mean accuracy of the pretrained features (`experiment_index.csv`, 0.439).
- [ ] Check all references in `report/latex/refs.bib` against the papers (written from memory).
- [ ] Add Qi et al. (2018) and He et al. (2022) to `report/latex/refs.bib` (cited in Related work). Check page numbers against CVPR open access and match the file's key style.
  ```bibtex
@inproceedings{qi2018imprinted,
  title     = {Low-Shot Learning with Imprinted Weights},
  author    = {Qi, Hang and Brown, Matthew and Lowe, David G.},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {5822--5830},
  year      = {2018}
}

@inproceedings{he2022mae,
  title     = {Masked Autoencoders Are Scalable Vision Learners},
  author    = {He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {16000--16009},
  year      = {2022}
}
```
- [ ] Figure 1 labels: add "LoRA r16 on qkv" for the PEFT model; confirm Task 2 used the prototype head and the 0.1 cosine margin.
- [ ] Parameter split in Section 3.7 against the PEFT run's printed `[params]` line.

**Content to check once drafted**

- [ ] Every number in the drafted sections matches the slides or a named run record.
- [ ] Which figures go in the paper, and which go to the appendix.

**At the end**

- [ ] Assign an owner to each section for the final pass.
- [ ] Move the final text into the ICLR LaTeX template (`report/latex/main.tex`).
- [ ] Check the main text fits in 10 pages.

---

## Open suggestions (not accepted at snapshot time)

1. (2026-11-10) Add: "This fits the data: most images have a short side of 300–400 px (Section 3.4), so at 448 px they are already enlarged, and a larger input only splits the same pixels into more patches without adding detail. / In both cases the high-resolution phase starts from a model that has already plateaued at 224 px, and the epochs left are too few to catch up. / , because the checkpoint was trained at 224 px and objects appear larger at test time than it expects (Touvron et al., 2019)".
2. (2026-11-10) Remove: "AdaLoRA was 2.7 pp below LoRA, so it was not run at full scale.". Add: "AdaLoRA was 2.7 pp below LoRA: it trains 39% slower, so it fits fewer steps into the same time budget. It was not run at full scale.".
3. (2026-11-10) Remove: "and the kNN gain holds (+0.93 vs +0.85).". Add: "and the kNN gain holds (+0.93 vs +0.85), as expected since its settings were tuned on the labelled holdout rather than on Phase 1.".
4. (2026-11-10) Remove: "The weight β was chosen on the other half of the classes.".
5. (2026-11-10) Add: "A learned linear re-ranker added only 0.20 pp on top of the chosen variant with zoom, so we kept the fixed score, which needs no training.".
6. (2026-11-10) Remove: "details in the table above". Add: "details in the kNN variants table, Section 5.6".
7. (2026-11-10) Remove: "Section 3.7". Add: "Section 3.5".
