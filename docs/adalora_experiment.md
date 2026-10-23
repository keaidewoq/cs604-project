# AdaLoRA subset sweep

Run `notebooks/colab_launcher_qq_adalora.ipynb` from `qq_lora_ablation` on an
A100. Install `requirements-adalora.txt`; PEFT is pinned to the tested 0.21.2 API.

The four arms are LoRA r24 and AdaLoRA initial rank 24 with target average ranks
8, 16 and 20. Each gets 12 training minutes, seed 42 only (48 minutes total).
All share the official augreg2 ViT-B/16, 448px arm E augmentation, batch64 x
accumulation4, cosine prototype head, trainable norms/biases/position/class
embeddings, optimizer and learning rates. The fixed 1,000-class subset uses
40 fitting and five holdout images/class, fitting-only prototypes and original
10,000-way labels. Evaluation is crop0.9 without flip or zoom.

```bash
python -m src.qq_adalora_ablation --preflight-only
python -m src.qq_adalora_ablation --memory-check-only
python -m src.qq_adalora_ablation --data_root /content/data --output_dir /content/drive/MyDrive/CS604_RUNS/qq_adalora_448_c1000_seed42_v1
```

## Algorithm and integration

`src/peft/adalora.py` attaches official PEFT `SVDLinear` layers to the same
qkv/proj/fc1/fc2 targets as LoRA. It uses PEFT `RankAllocator` for gradient-based
importance estimates, EMA sensitivity/uncertainty and global component masking.
The regularizer matches PEFT's average Frobenius penalty on A/B Gram matrices,
with coefficient 0.5. It is applied once per optimizer update after accumulating
classification gradients and before clipping. Allocation runs after the optimizer
update, while gradients still exist, and before clearing them.

The one schedule change is deliberate: the original cubic rank-budget schedule
uses elapsed **training time** rather than an estimated optimizer-step count.
The first 10% retains all components, the middle 70% reduces the total rank
budget with cubic decay (mask every 10 optimizer updates), and the final 20%
holds the selected mask fixed. Importance updates happen every optimizer step
until that final phase. Masked singular values are re-zeroed after each final
optimizer update. The LR warmup stays at 5%, independently of rank warmup.

All initially allocated backbone parameters, including singular values, count
toward the strict <5% cap. Matrices have different shapes, so an average rank
budget does not imply a fixed active parameter percentage. The final report
counts active components using their actual per-matrix costs plus extras.
Pruning masks do not resize A/B/E tensors or optimizer state. Tied importance
scores can prune more components than the requested budget, as in PEFT.

The memory probe exercises two synthetic optimizer updates, including
regularization, AdamW state, importance buffers and a final mask. It uses
target8 because all three targets allocate identical initial tensors. It
checks memory, not accuracy; real-data allocations can differ.

## Reading the results

Choose by final holdout accuracy after equal training time. `results.csv` also
records throughput, epochs, allocated parameters, actual final active rank/params
and whether the final mask was reached. Inspect `adalora_rank_log.csv` and
`adalora_final.json` in each AdaLoRA run. Both best/last checkpoints preserve
the trainable A/B/E tensors, head and extras plus rank-allocation metadata.
`ranknum` remains the deterministic initial rank and is recreated by the config.
Reload the same config and original pretrained backbone for prediction.

Use a new experiment folder after settings/code/hardware changes or interrupted
runs. Finished arms with matching fingerprints may be reused. This launcher
does not resume optimizer/importance history or produce official submissions.
Subset winners require full-data validation; close one-seed scores are inconclusive.

Sources: [PEFT AdaLoRA documentation](https://huggingface.co/docs/peft/en/package_reference/adalora),
[PEFT layers and allocator](https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/layer.py),
[PEFT training integration](https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/model.py),
[original implementation](https://github.com/QingruZhang/AdaLoRA).
