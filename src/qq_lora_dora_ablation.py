"""Equal-time LoRA/DoRA screen with the arm E recipe and a shared rank below 5%."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import shutil
from pathlib import Path

import torch

from src.model import build_model
from src.peft.lora import LoRALinear
from src.qq_lora_ablation import prepare_split, run_arm, save_results, write_json
from src.utils import Config, load_config


def plan_arms(base):
    """Count an untrained model; derive rank costs from its actual target shapes."""
    reference_rank = int(base.lora_rank)
    if reference_rank < 1:
        raise ValueError('Reference rank must be positive')
    cap = float(base.max_adaptation_pct)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('max_adaptation_pct must be finite and positive')
    cfg = Config({**base.to_dict(), 'pretrained': False, 'lora_dora': False,
                  'max_adaptation_pct': None})
    model = build_model(cfg)
    stats = model.adaptation_stats
    layers = [layer for layer in model.modules() if isinstance(layer, LoRALinear)]
    if not layers:
        raise ValueError('No LoRA target layers found')
    per_rank = sum(layer.linear.in_features + layer.linear.out_features for layer in layers)
    magnitudes = sum(layer.linear.out_features for layer in layers)
    extras = stats['adaptation_parameters'] - reference_rank * per_rank
    total = stats['original_backbone_parameters']

    def arm(rank, dora):
        count = extras + rank * per_rank + (magnitudes if dora else 0)
        return dict(arm=f'{"dora" if dora else "lora"}_r{rank}', rank=rank, dora=dora,
                    adaptation_parameters=count, original_backbone_parameters=total,
                    adaptation_pct=100.0 * count / total)

    reference = arm(reference_rank, False)
    if reference['adaptation_pct'] >= cap:
        raise ValueError('Reference LoRA exceeds the parameter cap')
    shared_rank = next((rank for rank in range(reference_rank, 0, -1)
                        if arm(rank, True)['adaptation_pct'] < cap), None)
    if shared_rank is None:
        raise ValueError('No positive shared rank fits DoRA below the parameter cap')
    arms = [reference]
    if shared_rank != reference_rank:
        arms.append(arm(shared_rank, False))
    arms.append(arm(shared_rank, True))
    del model
    gc.collect()
    return arms


def matched_epoch_results(results):
    """Compare logged evaluations at common complete epochs, excluding partials."""
    shared_rank = next(row['rank'] for row in results if row['dora'])
    pair = {row['dora']: row for row in results if row['rank'] == shared_rank}
    if len(pair) != 2:
        raise ValueError('Missing the shared-rank LoRA/DoRA pair')
    histories = {}
    for dora, result in pair.items():
        with (Path(result['run_dir']) / 'epoch_log.csv').open() as handle:
            histories[dora] = {int(row['epoch']): row for row in csv.DictReader(handle)
                               if row['epoch_complete'] == '1' and row['val_acc'] != ''}
    rows = []
    for epoch in sorted(histories[False].keys() & histories[True].keys()):
        lora, dora = histories[False][epoch], histories[True][epoch]
        rows.append(dict(epoch=epoch, lora_val_acc=float(lora['val_acc']),
                         dora_val_acc=float(dora['val_acc']),
                         delta_pp=100 * (float(dora['val_acc']) - float(lora['val_acc'])),
                         lora_val_loss=float(lora['val_loss']), dora_val_loss=float(dora['val_loss'])))
    return rows


def save_comparison(root, results, minimum_gain):
    comparisons = []
    for seed in sorted({row['seed'] for row in results}):
        seeded = [row for row in results if row['seed'] == seed]
        dora = next((row for row in seeded if row['dora']), None)
        if dora is None:
            continue
        lora = next(row for row in seeded if not row['dora'] and row['rank'] == dora['rank'])
        delta = dora['final_val_acc'] - lora['final_val_acc']
        comparisons.append(dict(seed=seed, shared_rank=dora['rank'], delta_pp=100 * delta,
                                speed_ratio=dora['images_per_second'] / lora['images_per_second'],
                                clears_screening_threshold=delta > minimum_gain,
                                matched_complete_epochs=matched_epoch_results(seeded)))
    confirmed = len(comparisons) >= 2 and all(row['clears_screening_threshold'] for row in comparisons)
    write_json(Path(root) / 'comparison.json', dict(
        metric='final_val_acc after equal training minutes', comparisons=comparisons,
        min_gain_pp=100 * minimum_gain, promising_across_seeds=confirmed,
        note='Subset screen, not statistical significance or a full-data result. '
             'Common-epoch metrics use time-based schedules at different LR positions; '
             'they do not isolate adaptation quality from scheduling.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/qq_lora_dora_ablation.yaml')
    parser.add_argument('--data_root')
    parser.add_argument('--output_dir')
    parser.add_argument('--minutes', type=float, default=12)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--classes', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--split_dir', default='/content/qq_lora_dora_data')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.minutes <= 144 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Use unique seeds and positive minutes per arm no greater than 144')
    base = load_config(args.config)
    base.update(dict(max_train_minutes=args.minutes, ablation_classes=args.classes,
                     num_workers=args.workers))
    if (base.model_name != 'vit_base_patch16_224.augreg2_in21k_ft_in1k' or
            base.method != 'lora' or not base.pretrained or base.get('prompt_mode') != 'none' or
            base.get('prompt_length', 0) or base.epochs is not None):
        raise ValueError('Use the official pretrained ViT, LoRA without prompts, and time-only training')
    arms = plan_arms(base)
    print('\nParameter preflight (classifier excluded; no pretrained weights downloaded):')
    for arm in arms:
        print(f"{arm['arm']}: {arm['adaptation_parameters']:,} / "
              f"{arm['original_backbone_parameters']:,} = {arm['adaptation_pct']:.6f}%")
    print(f'Total training budget: {len(arms) * len(args.seeds) * args.minutes:g} minutes; '
          'prototype extraction, evaluation and saving take additional wall time.')
    if args.preflight_only:
        return
    if not args.data_root or not args.output_dir:
        parser.error('--data_root and --output_dir are required for training')
    if not torch.cuda.is_available():
        raise RuntimeError('Select an A100 GPU runtime for training')
    properties = torch.cuda.get_device_properties(0)
    import timm
    base.comparison_hardware = dict(gpu=properties.name, memory=properties.total_memory,
                                    torch=torch.__version__, timm=timm.__version__, cuda=torch.version.cuda)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    split_cfg = Config({**base.to_dict(), 'seed': int(base.split_seed)})
    split_key = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    split = prepare_split(args.data_root, Path(args.split_dir) / split_key, split_cfg)
    recorded = root / 'split_manifests'
    recorded.mkdir(exist_ok=True)
    for name in ['train.txt', 'val.txt', 'split.json']:
        target = recorded / name
        if target.exists() and target.read_bytes() != (split / name).read_bytes():
            raise ValueError('Saved split differs; use a new output folder')
        shutil.copyfile(split / name, target)
    # Each arm also fingerprints resolved config, source code, split and dependencies.
    results = []
    for seed in args.seeds:
        seed_root = root / f'seed{seed}'
        seed_root.mkdir(exist_ok=True)
        for arm in arms:
            cfg = Config({**base.to_dict(), 'seed': seed, 'lora_rank': arm['rank'],
                          'lora_alpha': arm['rank'], 'lora_dora': arm['dora']})
            result = run_arm(cfg, split, seed_root, arm['arm'],
                             int(cfg.img_size), float(cfg.full_image_probability))
            if not math.isclose(result['adaptation_pct'], arm['adaptation_pct'], abs_tol=1e-9):
                raise ValueError('Runtime parameter count differs from preflight')
            row = {**result, 'seed': seed, 'rank': arm['rank'], 'dora': arm['dora']}
            results.append(row)
            save_results(root, results)
            save_comparison(root, results, float(base.ablation_min_gain))
        print(f"[seed {seed}] See comparison.json for shared-rank differences and common-epoch metrics")


if __name__ == '__main__':
    main()
