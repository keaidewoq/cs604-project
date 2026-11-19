# LoRA + VPT ablation

Branch: `qq_lora_vpt_ablation`. Launcher:
`notebooks/colab_lora-r24_336px_vpt-prompt-length-sweep.ipynb`.

This is an experiment implementation, not a recorded GPU result. All arms use
the existing `lora_r24_336px_qkv-proj-fc_extras_proto-cosine_2.4h.yaml` baseline: LoRA r24 on qkv/proj/fc1/fc2,
trainable norms/biases/positional embeddings/CLS, prototype cosine head, 336px,
LoRA/extras LR 3e-4, head LR 3e-3, weight decay 0.05, label smoothing 0.1,
gradient clipping 1.0 and effective batch 256. Prompt embeddings have their own
learning rate and no weight decay.

| Arm | Prompts | Added parameters | Backbone trainable % |
|---|---|---:|---:|
| A | None | 0 | 4.653365 |
| B | Shallow, 8 tokens | 6,144 | 4.660510 |
| C | Shallow, 16 tokens | 12,288 | 4.667655 |
| D | Shallow, 32 tokens | 24,576 | 4.681946 |
| E | Deep, 8 tokens at each of 12 blocks | 73,728 | 4.739108 |

Counts were checked by constructing the actual ViT-B/16 model without pretrained
weights. The denominator is the original 85,986,816-parameter backbone at 336;
the classifier is excluded and the cap includes all trained backbone extras.

Shallow prompts enter after CLS and before image patches, after the model's
positional embedding/dropout and Identity pre-norm. Deep prompts replace the
previous block's prompt outputs before each subsequent block. Prompt slots do
not accumulate. The model remains CLS-pooled. Hooks preserve existing block names
and allow trainable-only PEFT checkpoints and LoRA merging. Gradient checkpointing
is not supported by the prompt hooks.

## Run in Colab

1. Commit and push `qq_lora_vpt_ablation`, then open the notebook from that branch
   in Colab and choose an A100 GPU runtime.
2. The setup cell clones or pulls the branch into `/content/cs604_lora_vpt`.
   Private-repo authentication uses a hidden token passed through a child-process
   environment, not the remote URL. No source archive is required.
3. Mount Drive with `CS604_RUNS` available in My Drive. The launcher downloads
   the course data to `/content/data`.
4. Review the editable settings and execute the notebook in order. The default
   screen runs the prompt-LR pilot, then A/B/E/C/D. Optional repeats, final training,
   prediction and latency measurement are disabled initially.

The notebook runs small synthetic checks before the course experiments. No
course data or pretrained model download is required for those checks.

## Split, pilot and comparison

Default: 1,000 classes selected with split seed 42, 40 fitting and 5 holdout
images/class, five epochs per screening arm, training seed 42. Subset classes are
remapped to contiguous IDs with the original mapping saved. Set `CLASSES=10000`
for a 400k/50k all-class internal split. All arms within an experiment share the
same manifests and fitting-only prototypes. Existing all-data prototypes must
not be reused for screening because they include the holdout.

The three-epoch LR pilot uses B at prompt LR 3e-4 and 1e-3. Among rates within
0.3 percentage points of the best holdout score, select the lower rate. Every
prompted screen arm uses that same selected rate and restarts from pretrained
weights. The five-epoch schedule has 5% step warmup then cosine; there is no
training-time cutoff during screening. Neither pilot nor screening establishes
convergence or replaces a full-budget comparison.

`comparison.csv` includes best/final holdout accuracy, best epoch, final loss,
percentage-point improvement over A at the same seed, trainable backbone count,
prompt count, peak allocated GiB, training-only minutes, wall GPU-hours, training
throughput and the exact run directory. The notebook writes learning curves.
Peak VRAM is allocated GPU memory, not reserved memory or total device usage.
Pilot and screening use the same development holdout; reported selection scores
are not an independent estimate of generalization.

Optional seed repeats compare A with `FINALIST_ARM` at seeds 42/43/44, reuse
matching completed seed-42 runs, and write `comparison_repeats.csv`. They vary
training randomness while keeping the split fixed. The notebook displays means,
standard deviations and paired accuracy differences.

## Completion and provenance

Artifacts are under `CS604_RUNS/<EXPERIMENT_NAME>/`: split manifests, source and
manifest hashes, class mapping, prototype provenance, LR selection and separate
pilot/screen run directories. Each completed run has a completion marker with
its settings and artifact hashes. Matching complete runs are skipped; changed
settings or artifacts cause an error. Use a new experiment name when changing
the recipe, split, hardware/software context or source version. An interrupted
arm restarts from pretrained weights rather than resuming optimizer state.

Keep batch size and accumulation identical across arms. If memory requires
changing 128 x 2 to 64 x 4, apply it to every arm in a new experiment.

## Final training and inference

After choosing an arm and prompt LR, enable the optional final stage. It requires
the original IDs 0..9999 and 45 training images/class, rebuilds all-data prototypes,
starts from pretrained weights and trains for 144 training-only minutes with
5% time warmup and cosine decay. There is no epoch ceiling. Setup, prototype
extraction and checkpoint saving are outside this training timer. Final training
uses a separate directory and never loads a screening checkpoint.

Predictions use the saved final config and `last.pt`. Default crop 0.9 without
flip allows matched single-view comparisons. Crop/flip can be changed explicitly;
the repository does not contain Van's zoom code, so the historical 79.9% LoRA
zoom result is a reference, not the matched baseline for this experiment.

Optional latency measurement loads the trained checkpoint, verifies FP32 logits
before/after merging LoRA, then measures single-view BF16 forwards at batch sizes
1 and 64. Prompts remain active after merging. Timings exclude image loading,
preprocessing, transfers and TTA. Compare on the same GPU and measurement settings.

## CLI and verification

```bash
python -m src.lora_vpt_ablation screen \
  --data-root /content/data \
  --output-dir /content/drive/MyDrive/CS604_RUNS/qq_lora_vpt_336_c1000_seed42_v1 \
  --lr-pilot --arms A B E C D

python -m src.lora_vpt_ablation final \
  --data-root /content/data \
  --output-dir /content/drive/MyDrive/CS604_RUNS/qq_lora_vpt_final_B \
  --arm B --prompt-lr 0.0003 --minutes 144

python -m pytest tests/test_lora_vpt.py -q
```

Use the prompt LR actually selected by your pilot for final training. Tests cover
token placement/replacement, gradients, initial weight and RNG consistency,
optimizer membership, parameter cap, checkpoint reload and merging, split
separation/remapping, LR selection, screen orchestration, synthetic training,
completed-run reuse and rejection of a subset as final data.
