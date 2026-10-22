"""Equal-time LoRA crop/resolution experiments; run via the Qian Colab launcher."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import ImageListDataset, build_transforms
from src.model import build_model
from src.train import amp_context, train
from src.utils import Config, load_config, save_config, set_seed


ARMS = (("A", 336, 0.0), ("B", 336, 0.25), ("C", 384, 0.0), ("D", 384, 0.25))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def partition_rows(rows, n_classes, holdout, seed):
    """Keep original 10k class IDs; choose classes/holdout once for every arm."""
    by_class = defaultdict(list)
    for path, label in rows:
        by_class[label].append(path)
    if not 0 < n_classes <= len(by_class) or holdout < 1:
        raise ValueError("Choose a positive class count within the dataset and holdout >= 1")
    rng = random.Random(seed)
    classes = sorted(rng.sample(sorted(by_class), n_classes))
    fit, val = [], []
    for label in classes:
        paths = sorted(by_class[label])
        if len(paths) <= holdout or len(paths) != len(set(paths)):
            raise ValueError(f"Class {label} has too few or repeated image paths")
        rng.shuffle(paths)
        val.extend((path, label) for path in paths[:holdout])
        fit.extend((path, label) for path in paths[holdout:])
    if {path for path, _ in fit} & {path for path, _ in val}:
        raise ValueError("Fit and holdout image paths overlap")
    return fit, val


def prepare_split(data_root, experiment_root, cfg):
    source = Path(data_root).resolve()
    split = Path(experiment_root) / "split"
    spec = dict(source=str(source), source_sha256=file_hash(source / "train.txt"),
                classes=int(cfg.ablation_classes), holdout=int(cfg.ablation_holdout), seed=int(cfg.seed))
    rows = []
    for line in (source / "train.txt").read_text().splitlines():
        if line.strip():
            path, label = line.split()
            rows.append((path, int(label)))
    fit, val = partition_rows(rows, spec['classes'], spec['holdout'], spec['seed'])
    contents = {name: ''.join(f'{p} {y}\n' for p, y in items)
                for name, items in [('train.txt', fit), ('val.txt', val)]}
    marker = split / "split.json"
    if marker.exists():
        if json.loads(marker.read_text()) != spec or any((split / name).read_text() != text for name, text in contents.items()):
            raise ValueError("Saved split differs. Use a new experiment name/output folder.")
        return split
    split.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        (split / name).write_text(text)
    for entry in source.iterdir():
        if entry.name in ("train.txt", "val.txt", "test.txt"):
            continue
        link = split / entry.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(entry, target_is_directory=entry.is_dir())
    write_json(marker, spec)
    print(f"[split] {len(fit)} fitting / {len(val)} holdout images; head remains {cfg.num_classes}-way")
    return split


@torch.no_grad()
def ensure_prototypes(cfg, split, experiment_root):
    """Cache per-resolution fit-only means; unused classes get seeded random rows."""
    path = Path(experiment_root) / f"prototypes_fit_{cfg.img_size}.pt"
    metadata = path.with_suffix('.json')
    prototype_seed = int(cfg.get('prototype_seed', cfg.seed))
    spec = dict(model_name=cfg.model_name, img_size=int(cfg.img_size), num_classes=int(cfg.num_classes),
                fit_sha256=file_hash(split / 'train.txt'), seed=prototype_seed, eval_crop_pct=cfg.eval_crop_pct)
    if path.exists():
        if not metadata.exists() or json.loads(metadata.read_text()) != spec:
            raise ValueError(f"Prototype metadata mismatch: {path}. Use a new experiment folder.")
        return path
    set_seed(prototype_seed)
    proto_cfg = Config({**cfg.to_dict(), 'method': 'linear_probe', 'head_init': 'random',
                        'prompt_mode': 'none', 'prompt_length': 0, 'full_image_probability': 0.0})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(proto_cfg).to(device).eval()
    _, transform = build_transforms(proto_cfg)
    dataset = ImageListDataset(split, 'train', transform)
    loader = DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=False,
                        num_workers=int(cfg.num_workers), pin_memory=True)
    sums = torch.zeros(int(cfg.num_classes), model.num_features, device=device)
    counts = torch.zeros(int(cfg.num_classes), device=device)
    for images, labels, _ in tqdm(loader, desc=f'Fit-only prototypes {cfg.img_size}'):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with amp_context(device, bool(cfg.bf16)):
            features = model.forward_head(model.forward_features(images), pre_logits=True)
        sums.index_add_(0, labels, F.normalize(features.float(), dim=1))
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    selected = sorted({label for _, label in dataset.samples})
    if not selected or not (counts[selected] > 0).all():
        raise ValueError('Missing fitting examples for a selected class')
    # Do not build unseen-class means from the holdout or full-data prototypes.
    generator = torch.Generator().manual_seed(prototype_seed)
    prototypes = F.normalize(torch.randn(int(cfg.num_classes), model.num_features, generator=generator), dim=1)
    prototypes[selected] = F.normalize(sums[selected] / counts[selected, None], dim=1).cpu()
    temporary = path.with_suffix('.pt.tmp')
    torch.save(dict(prototypes=prototypes, samples_per_class=counts.cpu(), **spec), temporary)
    temporary.replace(path)
    write_json(metadata, spec)
    del model, loader, sums, counts
    gc.collect()
    torch.cuda.empty_cache()
    return path


def summarize(run_dir):
    run_dir = Path(run_dir)
    with (run_dir / 'epoch_log.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row['val_acc'] != '']
    if not rows:
        raise ValueError(f'No labeled holdout evaluation in {run_dir}')
    stats = dict(line.split('=', 1) for line in (run_dir / 'run_stats.txt').read_text().splitlines())
    seconds = float(stats['training_seconds'])
    return dict(final_val_acc=float(rows[-1]['val_acc']), final_val_loss=float(rows[-1]['val_loss']),
                best_val_acc=max(float(row['val_acc']) for row in rows),
                training_minutes=seconds / 60, wall_minutes=float(stats['wall_s']) / 60,
                images_per_second=sum(int(row['images_processed']) for row in rows) / seconds,
                epochs=sum(int(row['batches_processed']) / int(row['batches_total']) for row in rows),
                peak_vram_gib=float(stats['peak_vram_gb']), adaptation_pct=float(stats['adaptation_pct']),
                run_dir=str(run_dir))


def candidate_384(results, minimum_gain):
    best336 = max((row for row in results if row['resolution'] == 336), key=lambda r: r['final_val_acc'])
    best384 = max((row for row in results if row['resolution'] == 384), key=lambda r: r['final_val_acc'])
    return best384 if best384['final_val_acc'] > best336['final_val_acc'] + minimum_gain else None


def run_arm(base, split, experiment_root, name, resolution, probability):
    cfg = Config(base.to_dict())
    cfg.update(dict(img_size=resolution, full_image_probability=probability, data_root=str(split),
                    output_dir=str(Path(experiment_root) / name),
                    prototype_file=str(Path(experiment_root) / f'prototypes_fit_{resolution}.pt')))
    spec = cfg.to_dict()
    spec['fit_sha256'] = file_hash(split / 'train.txt')
    spec['holdout_sha256'] = file_hash(split / 'val.txt')
    # Reject silent reuse after code/dependency changes affecting the experiment.
    import timm
    spec['versions'] = dict(torch=torch.__version__, timm=timm.__version__)
    spec['code'] = {str(path): file_hash(path) for path in sorted(Path('src').rglob('*.py'))}
    spec['code']['configs/qq_lora_ablation.yaml'] = file_hash('configs/qq_lora_ablation.yaml')
    arm_root = Path(cfg.output_dir)
    result_path = arm_root / 'result.json'
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved['spec'] != spec or not (Path(saved['result']['run_dir']) / 'last.pt').is_file():
            raise ValueError(f'Saved arm {name} differs or checkpoint missing. Use a new experiment folder.')
        print(f'[reuse] {name}: {saved["result"]["final_val_acc"]:.4f}')
        return saved['result']
    if arm_root.exists() and any(arm_root.iterdir()):
        raise ValueError(f'Incomplete arm {name} exists. Choose a new experiment folder; partial runs are not silently reused.')
    prototype_path = ensure_prototypes(cfg, split, experiment_root)
    cfg.prototype_file = str(prototype_path)
    arm_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, arm_root / 'resolved.yaml')
    run_dir = train(cfg)
    result = dict(arm=name, resolution=resolution, full_image_probability=probability, **summarize(run_dir))
    write_json(result_path, dict(spec=spec, result=result))
    gc.collect()
    torch.cuda.empty_cache()
    return result


def save_results(root, results):
    root = Path(root)
    with (root / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_json(root / 'results.json', results)
    print('\nFinal holdout scores (same training minutes; no zoom/flip):')
    for row in results:
        print(f"{row['arm']}: {row['resolution']} pad={row['full_image_probability']:.0%} "
              f"acc={row['final_val_acc']:.4f} epochs={row['epochs']:.2f} "
              f"speed={row['images_per_second']:.1f} img/s VRAM={row['peak_vram_gib']:.2f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/qq_lora_ablation.yaml')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--minutes', type=float, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--classes', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stage', choices=['main', '448'], default='main')
    parser.add_argument('--split_dir', default='/content/qq_lora_ablation_data')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime; these experiments train LoRA. A100 is recommended.')
    if not 0 < args.minutes <= 144:
        raise ValueError('Training minutes per arm must be positive and no greater than 144')
    base = load_config(args.config)
    base.update(dict(max_train_minutes=args.minutes, seed=args.seed,
                     ablation_classes=args.classes, num_workers=args.workers))
    if base.model_name != 'vit_base_patch16_224.augreg2_in21k_ft_in1k' or base.method != 'lora':
        raise ValueError('This experiment requires the official pretrained ViT and LoRA recipe')
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep image symlinks on local disk: Google Drive does not support them.
    split_key = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    split = prepare_split(args.data_root, Path(args.split_dir) / split_key, base)
    recorded = root / 'split_manifests'
    recorded.mkdir(exist_ok=True)
    for name in ['train.txt', 'val.txt', 'split.json']:
        shutil.copyfile(split / name, recorded / name)
    if args.stage == '448' and not all((root / name / 'result.json').exists() for name, _, _ in ARMS):
        raise ValueError('Complete A–D first; the optional cell never launches missing main arms')
    results = [run_arm(base, split, root, *arm) for arm in ARMS]
    pick = candidate_384(results, float(base.ablation_min_gain))
    write_json(root / 'selection.json', dict(candidate_384=pick, min_gain=float(base.ablation_min_gain),
                                            metric='final_val_acc', note='One-seed subset screen; confirm before full-data training.'))
    if args.stage == '448':
        if pick is None:
            print('384 did not beat the best 336 arm by more than 0.3 points; 448 skipped.')
        else:
            matched = next(row for row in results if row['resolution'] == 336 and row['full_image_probability'] == pick['full_image_probability'])
            print(f"384 winner {pick['arm']}; matched 336 accuracy={matched['final_val_acc']:.4f}. Reusing the completed 384 run.")
            results.append(run_arm(base, split, root, 'E_448', 448, pick['full_image_probability']))
    save_results(root, results)


if __name__ == '__main__':
    main()
