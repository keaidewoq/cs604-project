# Zoom at 448 px works best from a crop 0.9 view

Author: Van. Written 2026-10-20. Prediction only: nothing is retrained.

**Short version.** At 448 px, predict with `src.predict --crop_pct 0.9 --zoom` instead of
`--crop_pct 0.75 --zoom`. It added **+0.5** on CodaBench for the AdaptFormer final (0.809 ->
0.814) and **+0.4** for the full-FT final (0.828 -> 0.832), for the same two forward passes.

## Why we looked

`--zoom` (`src/van_zoom.py`) classifies the image, finds the region the model used (a
class-matched saliency map), cuts that region from a 2x-resolution copy of the same view,
classifies it again and averages the two softmaxes. Its settings (eval crop 0.75, box mass
0.7, minimum box 8 patches) were tuned at 336 px. At 448 the prediction-time views added
less on CodaBench:

| Run | Plain | Crop 0.75 + flip | Crop 0.75 + zoom |
| --- | --- | --- | --- |
| AdaptFormer 384 | 0.794 | - | 0.807 (+1.3) |
| AdaptFormer 448 | 0.804 | 0.806 (+0.2) | 0.809 (+0.5) |
| Full FT 448 | 0.825 | 0.827 (+0.2) | 0.828 (+0.3) |

## How it was tuned

Evaluation only, on Van's fixed internal holdout: 5 training images per class (50,000
images) that the screening model never trained on. The model was Van's 448 px AdaptFormer
screening run (same recipe as the final, trained on 10 images per class; 62.7% top-1). It is
weaker than the final, so its gains are larger than CodaBench's, but it ranks the settings.

Swept: zoom crop 0.75 / 0.9 / 1.0, box mass 0.5 / 0.7 / 0.9, minimum box 8 / 11 patches
(11 patches at 448 is the same share of the image as 8 at 336). To avoid picking a lucky
setting, the best one was **chosen on the even-numbered half of the holdout and reported
on the odd half**.

Top-1 (%), plain + zoom (2 passes), even half / odd half:

| Zoom crop / min box | Mass 0.5 | **Mass 0.7** | Mass 0.9 |
| --- | --- | --- | --- |
| 0.75 / 8 (old setting) | 63.580 / 62.796 | 63.552 / **63.244** | 63.684 / 63.228 |
| 0.75 / 11 | 63.632 / 62.840 | 63.556 / 63.244 | 63.684 / 63.228 |
| **0.9 / 8** | 63.984 / 63.288 | **64.116 / 63.520** | 63.868 / 63.172 |
| 1.0 / 8 | 63.952 / 63.408 | 63.996 / 63.460 | 63.648 / 62.916 |
| 0.9 / 8, + both flips (4 passes) | 64.072 / 63.408 | 64.224 / 63.568 | 64.024 / 63.176 |

Without zoom (even / odd): crop 0.75 plain 63.192 / 62.612, + flip 63.316 / 62.780;
crop 0.9 plain 63.048 / 62.384, + flip 63.280 / 62.492.

## What it shows

- **Crop 0.9 with mass 0.7 wins** (best on the even half): +0.28 over the old setting on the
  odd half (paired z 1.7). On CodaBench it gave more: +0.5 for AdaptFormer, +0.4 for full FT.
- **Why:** the tighter 0.75 crop helps a single plain view (+0.2) but hurts the zoom. At 448
  the zoom already enlarges the subject; cropping first removes context that the zoom box
  needs. Van's full-FT resolution screen saw the same thing: the gain from crop 0.75 over 0.9
  disappeared at 448.
- **Keep mass 0.7.** The minimum box (8 vs 11 patches) makes almost no difference: the
  0.7-mass box is rarely smaller than 11 patches.
- **Zoom beats flip** for the same two passes (+0.7 on the odd half), so flip is not used in
  the submission. Adding both flips to the zoom (4 passes) adds only +0.05: not worth double
  the cost.
- The setting was chosen on an AdaptFormer model and carried over to full FT unchanged.

## How to use it

```bash
python -m src.predict --config configs/<your_448_config>.yaml --ckpt <run>/last.pt \
    --split val --team 08 --crop_pct 0.9 --zoom
```

The zoom builds 896 px copies of every image at 448, which needs a lot of host memory. On a
16 GB laptop GPU with 32 GB RAM, `--batch_size 32` with `num_workers: 4` worked, and the
default batch with 12 workers crashed part-way. The Colab zoom prediction for the
AdaptFormer 448 run also stopped part-way (31,062 of 50,000 rows): **check the row count
before submitting.**

CodaBench IDs: AdaptFormer 448 crop 0.9 + zoom 1002322 (0.814), crop 0.75 + zoom 1001880
(0.809); full FT 448 crop 0.9 + zoom 1004167 (0.832), crop 0.75 + zoom 995951 (0.828).
