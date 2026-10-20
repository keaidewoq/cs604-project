"""Predict with Van's autozoom, independently of the standard predictor.

python scripts/predict_zoom.py --config /path/to/run/config.yaml \
    --ckpt /path/to/run/last.pt --split val --team 08 --out_dir /path/to/predictions

Defaults: crop 0.75, zoom mass 0.7, two passes, no flip. Both passes use the
same model; the 2x-resolution companion image is only used to extract the crop.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import van_zoom
from src.data import ImageListDataset
from src.model import build_model
from src.predict import load_checkpoint
from src.utils import apply_overrides, get_device, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--team', required=True)
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--out_dir', default='.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--crop_pct', type=float, default=0.75)
    parser.add_argument('--zoom_mass', type=float, default=van_zoom.DEFAULT_MASS)
    parser.add_argument('--tta_flip', action='store_true',
                        help='Also predict flipped versions of both views (four passes).')
    return parser.parse_args()


@torch.no_grad()
def predict_batch(model, batch, device, use_bf16, zoom_mass, tta_flip=False):
    images, big = (view.to(device, non_blocking=True) for view in batch)
    enabled = bool(use_bf16) and device.type == 'cuda'
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
        tokens = model.forward_features(images)
        first_logits = model.forward_head(tokens).float()
        close_up = van_zoom.zoom_view(model, tokens, first_logits, big, mass=zoom_mass)
        views = [close_up]
        if tta_flip:
            views += [torch.flip(images, dims=[3]), torch.flip(close_up, dims=[3])]
        probabilities = [first_logits.softmax(dim=1)]
        for view in views:
            probabilities.append(model(view).float().softmax(dim=1))
    return torch.stack(probabilities).mean(dim=0)


@torch.no_grad()
def run_predict(cfg, ckpt_path, split, team, out_dir, crop_pct=0.75,
                zoom_mass=van_zoom.DEFAULT_MASS, tta_flip=False):
    if not math.isfinite(crop_pct) or not 0 < crop_pct <= 1:
        raise ValueError('crop_pct must be in (0, 1]')
    if not math.isfinite(zoom_mass) or not 0 < zoom_mass <= 1:
        raise ValueError('zoom_mass must be in (0, 1]')
    if int(cfg.batch_size) < 1 or int(cfg.num_workers) < 0:
        raise ValueError('Choose positive batch_size and nonnegative num_workers')
    device = get_device()
    transform = van_zoom.two_view_transform(cfg, crop_pct)
    dataset = ImageListDataset(cfg.data_root, split, transform=transform)
    loader = DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=False,
                        num_workers=int(cfg.num_workers), pin_memory=device.type == 'cuda')
    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{split}_team_{team}.txt'
    with out_path.open('w', encoding='utf-8') as handle:
        for batch, _, paths in tqdm(loader, desc=f'predict {split} + zoom'):
            probabilities = predict_batch(model, batch, device, bool(cfg.bf16), zoom_mass, tta_flip)
            predictions = probabilities.argmax(dim=1).cpu().tolist()
            for path, prediction in zip(paths, predictions):
                handle.write(f'{path} {prediction}\n')
    print(f'[zoom] wrote {out_path} ({len(dataset)} lines; crop_pct={crop_pct}, '
          f'zoom_mass={zoom_mass}, tta_flip={tta_flip})')
    return out_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, data_root=args.data_root, batch_size=args.batch_size,
                    num_workers=args.num_workers)
    run_predict(cfg, args.ckpt, args.split, args.team, args.out_dir,
                crop_pct=args.crop_pct, zoom_mass=args.zoom_mass, tta_flip=args.tta_flip)


if __name__ == '__main__':
    main()
