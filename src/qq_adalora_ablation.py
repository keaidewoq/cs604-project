"""LoRA r24 vs PEFT AdaLoRA target8/16/20 at 448px; seed42, 12 minutes per arm."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import shutil
from pathlib import Path

import torch

from src.model import build_model
from src.qq_lora_ablation import file_hash, prepare_split, run_arm, save_results, write_json
from src.qq_lora_dora_ablation import check_training_memory
from src.utils import Config, load_config


def arm_config(base, arm):
    cfg = Config(base.to_dict())
    cfg.update(dict(method=arm['method'], lora_rank=24, lora_alpha=24, lora_dora=False,
                    adalora_init_rank=24, adalora_target_rank=arm['target_rank'] or 16))
    return cfg


def plan_sweep(base):
    if (int(base.lora_rank) != 24 or int(base.adalora_init_rank) != 24 or
            list(base.adalora_target_ranks) != [8, 16, 20]):
        raise ValueError('This sweep compares LoRA r24 and AdaLoRA init24 -> target8/16/20')
    arms = [dict(arm='lora_r24', method='lora', target_rank=0)]
    arms += [dict(arm=f'adalora_t{rank}', method='adalora', target_rank=rank)
             for rank in base.adalora_target_ranks]
    for arm in arms:
        cfg = arm_config(base, arm)
        cfg.pretrained = False
        model = build_model(cfg)
        arm.update(model.adaptation_stats)
        if hasattr(model, 'adalora_controller'):
            arm['initial_components'] = model.adalora_controller.initial_budget
            arm['target_components'] = model.adalora_controller.target_budget
        else:
            arm['initial_components'] = arm['target_components'] = 24 * 48
        del model
        gc.collect()
    return arms


def save_sweep_summary(root, results):
    baseline = next(row for row in results if row['method'] == 'lora')
    comparisons = [dict(arm=row['arm'], target_rank=row['target_rank'],
                        final_val_acc=row['final_val_acc'],
                        delta_pp=100 * (row['final_val_acc'] - baseline['final_val_acc']),
                        speed_ratio=row['images_per_second'] / baseline['images_per_second'],
                        allocation_frozen=row['allocation_frozen'],
                        final_average_rank=row['final_average_rank'],
                        final_active_pct=row['final_active_pct'])
                   for row in results if row['method'] == 'adalora']
    complete = len(comparisons) == 3
    allocated = complete and all(row['allocation_frozen'] for row in comparisons)
    best = max(comparisons, key=lambda row: row['final_val_acc']) if allocated else None
    write_json(Path(root) / 'comparison.json', dict(
        baseline='lora_r24', metric='final holdout accuracy after equal training minutes',
        comparisons=comparisons, complete=complete, allocation_complete=allocated, best_adalora=best,
        note='Seed42 subset screen, not CodaBench. Final active parameter counts include extras; '
             'allocated trainable tensors retain initial rank 24. Close scores are inconclusive.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/qq_adalora_ablation.yaml')
    parser.add_argument('--data_root')
    parser.add_argument('--output_dir')
    parser.add_argument('--minutes', type=float, default=12)
    parser.add_argument('--classes', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--split_dir', default='/content/qq_adalora_data')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--memory-check-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.minutes <= 144 or args.classes < 1 or args.workers < 0:
        raise ValueError('Use positive minutes <=144, positive classes and nonnegative workers')
    base = load_config(args.config)
    base.update(dict(max_train_minutes=args.minutes, ablation_classes=args.classes, num_workers=args.workers))
    if (base.model_name != 'vit_base_patch16_224.augreg2_in21k_ft_in1k' or not base.pretrained or
            int(base.img_size) != 448 or int(base.seed) != 42 or int(base.split_seed) != 42 or
            int(base.prototype_seed) != 42 or base.epochs is not None or
            base.prompt_mode != 'none' or base.get('prompt_length', 0) or
            not math.isclose(float(base.full_image_probability), .25) or
            int(base.batch_size) * int(base.grad_accum_steps) != 256):
        raise ValueError('Keep the official 448px arm E recipe, fixed seed42, no prompts and effective batch256')
    base.comparison_versions = {name: str(importlib.metadata.version(name))
                                for name in ['torch', 'timm', 'peft', 'transformers']}
    arms = plan_sweep(base)
    print('\nAllocated trainable backbone parameters, including SVD singular values; head excluded:')
    for arm in arms:
        print(f"{arm['arm']}: {arm['adaptation_parameters']:,} / "
              f"{arm['original_backbone_parameters']:,} = {arm['adaptation_pct']:.6f}%; "
              f"rank components {arm['initial_components']} -> {arm['target_components']}")
    print(f'Four arms x {args.minutes:g} minutes = {4 * args.minutes:g} training minutes, seed42 only. '
          'Setup, memory check, prototypes, validation and saving take additional wall time.')
    if args.preflight_only:
        return
    if not args.memory_check_only and (not args.data_root or not args.output_dir):
        parser.error('--data_root and --output_dir are required for timed training')
    if not torch.cuda.is_available():
        raise RuntimeError('Select an A100 GPU runtime')
    gpu = torch.cuda.get_device_properties(0)
    base.comparison_hardware = dict(gpu=str(gpu.name), memory=int(gpu.total_memory),
                                    cuda=str(torch.version.cuda), **base.comparison_versions)
    # All AdaLoRA arms allocate the same tensors; target8 also exercises masking.
    probe = arms[1]
    check_training_memory(arm_config(base, probe), dict(arm=probe['arm'], rank=24, dora=False))
    if args.memory_check_only:
        return
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    split_key = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    split = prepare_split(args.data_root, Path(args.split_dir) / split_key, base)
    recorded = root / 'split_manifests'
    recorded.mkdir(exist_ok=True)
    for name in ['train.txt', 'val.txt', 'split.json']:
        target = recorded / name
        if target.exists() and target.read_bytes() != (split / name).read_bytes():
            raise ValueError('Saved split differs; use a new experiment folder')
        shutil.copyfile(split / name, target)
    manifest = dict(config=base.to_dict(), arms=arms,
                    code={str(path): file_hash(path) for path in sorted(Path('src').rglob('*.py'))})
    plan_path = root / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != manifest:
        raise ValueError('Experiment settings/code changed; use a new experiment folder')
    write_json(plan_path, manifest)
    seed_root = root / 'seed42'
    seed_root.mkdir(exist_ok=True)
    results = []
    for arm in arms:
        cfg = arm_config(base, arm)
        result = run_arm(cfg, split, seed_root, arm['arm'], 448, .25)
        if not math.isclose(result['adaptation_pct'], arm['adaptation_pct'], abs_tol=1e-9):
            raise ValueError('Runtime parameter count differs from preflight')
        final = None
        if arm['method'] == 'adalora':
            final = json.loads((Path(result['run_dir']) / 'adalora_final.json').read_text())
        row = dict(**result, seed=42, method=arm['method'], target_rank=arm['target_rank'],
                   final_average_rank=final['average_active_rank'] if final else 24.,
                   final_active_pct=final['active_adaptation_pct'] if final else result['adaptation_pct'],
                   allocation_frozen=final['frozen'] if final else True)
        results.append(row)
        save_results(root, results)
        save_sweep_summary(root, results)
    print('Done. See comparison.json, per-arm adalora_rank_log.csv and adalora_final.json.')


if __name__ == '__main__':
    main()
