"""Write submission files: python -m src.predict --config ... --ckpt ... --split val --team xx"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from src.data import ImageListDataset, build_transforms
from src.model import build_model
from src.utils import apply_overrides, get_device, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Run inference and write a team submission file.")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="Path to best.pt / last.pt")
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--team", required=True, help="Team id used in the filename, e.g. xx or 01")
    p.add_argument("--data_root", default=None)
    p.add_argument("--out_dir", default=".")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--crop_pct", type=float, default=None,
                   help="Eval center-crop fraction (default: the model's own, 0.9). Off unless given.")
    p.add_argument("--tta_flip", action="store_true",
                   help="Average the softmax of each image and its horizontal flip. Off unless given.")
    p.add_argument("--zoom", action="store_true",
                   help="Van: add a zoomed view of the region the model used (src/van_zoom.py) and "
                        "average the softmax; with --tta_flip the zoom is flipped too. Off unless given.")
    p.add_argument("--zoom_mass", type=float, default=None,
                   help="Share of the saliency map the zoom box holds (default 0.7).")
    p.add_argument("--balance_per_class", type=int, default=None,
                   help="Van: Sinkhorn-balance the predictions so each class gets about this many "
                        "images (src/van_balance.py; val 5, test 10). Off unless given.")
    p.add_argument("--balance_iters", type=int, default=None,
                   help="Sinkhorn iterations for --balance_per_class (default 50).")
    return p.parse_args()


def load_checkpoint(model, ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    trainable_only = bool(ckpt.get("trainable_only", False)) if isinstance(ckpt, dict) else False
    missing, unexpected = model.load_state_dict(state, strict=not trainable_only)
    if missing:
        print(f"[predict] missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[predict] unexpected keys ({len(unexpected)}): {unexpected[:8]}")
    return ckpt


@torch.no_grad()
def run_predict(cfg, ckpt_path, split, team, out_dir, crop_pct=None, tta_flip=False,
                zoom=False, zoom_mass=None, balance_per_class=None, balance_iters=None):
    device = get_device()
    _, eval_tf = build_transforms(cfg, eval_crop_pct=crop_pct)
    if zoom:
        from src import van_zoom

        eval_tf = van_zoom.two_view_transform(cfg, crop_pct)
        zoom_mass = van_zoom.DEFAULT_MASS if zoom_mass is None else float(zoom_mass)
    dataset = ImageListDataset(cfg.data_root, split, transform=eval_tf)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_team_{team}.txt"

    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    balance = balance_per_class is not None
    if balance:
        from src import van_balance

        all_paths, top_p, top_i = [], [], []
    with out_path.open("w", encoding="utf-8") as f:
        for images, _, paths in tqdm(loader, desc=f"predict {split}"):
            if zoom:
                probs = _predict_zoom(model, images, device, amp_dtype, use_bf16, tta_flip, zoom_mass)
                if balance:
                    p, i = van_balance.topk(probs)
                    all_paths += list(paths); top_p.append(p); top_i.append(i)
                    continue
                preds = probs.argmax(dim=1).cpu().tolist()
                for rel_path, pred in zip(paths, preds):
                    f.write(f"{rel_path} {pred}\n")
                continue
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                logits = model(images)
                if tta_flip:
                    flipped = model(torch.flip(images, dims=[3]))
            if tta_flip:
                logits = logits.float().softmax(dim=1) + flipped.float().softmax(dim=1)
            if balance:
                p, i = van_balance.topk(logits.float().softmax(dim=1))
                all_paths += list(paths); top_p.append(p); top_i.append(i)
                continue
            preds = logits.argmax(dim=1).cpu().tolist()
            for rel_path, pred in zip(paths, preds):
                f.write(f"{rel_path} {pred}\n")

        if balance:
            top_p, top_i = torch.cat(top_p), torch.cat(top_i)
            top_p = top_p / top_p.sum(1, keepdim=True)
            iters = van_balance.DEFAULT_ITERS if balance_iters is None else int(balance_iters)
            num_classes = int(model.get_classifier().weight.shape[0])
            preds, _ = van_balance.sinkhorn(top_p, top_i, num_classes, balance_per_class, iters)
            for rel_path, pred in zip(all_paths, preds.tolist()):
                f.write(f"{rel_path} {pred}\n")
            torch.save({"paths": all_paths, "p": top_p.half(), "i": top_i.int()},
                       out_dir / f"{split}_team_{team}_top{van_balance.TOPK}.pt")
            counts = torch.bincount(preds, minlength=num_classes)
            changed = (preds != top_i[:, 0]).sum().item()
            print(f"[predict] balanced to {balance_per_class}/class ({iters} iters): {changed} of "
                  f"{len(preds)} predictions changed; per-class count max {counts.max().item()}, "
                  f"classes unused {(counts == 0).sum().item()}")

    zoom_note = f", zoom mass {zoom_mass}" if zoom else ""
    if balance:
        zoom_note += f", balanced {balance_per_class}/class"
    print(f"[predict] wrote {out_path} ({len(dataset)} lines; crop_pct={crop_pct or 'model default'}, "
          f"tta_flip={tta_flip}{zoom_note})")
    return out_path


def _predict_zoom(model, batch, device, amp_dtype, use_bf16, tta_flip, zoom_mass):
    """Plain view (+ flip) + zoomed view (+ its flip), softmax averaged (src/van_zoom.py)."""
    from src import van_zoom

    images = batch[0].to(device, non_blocking=True)
    big = batch[1].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
        tokens = model.forward_features(images)
        logits = model.forward_head(tokens).float()
        views = [van_zoom.zoom_view(model, tokens, logits, big, mass=zoom_mass)]
        if tta_flip:
            views += [torch.flip(images, dims=[3]), torch.flip(views[0], dims=[3])]
        probs = logits.softmax(dim=1)
        for view in views:
            probs = probs + model(view).float().softmax(dim=1)
    return probs


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, data_root=args.data_root, batch_size=args.batch_size)
    run_predict(cfg, args.ckpt, args.split, args.team, args.out_dir,
                crop_pct=args.crop_pct, tta_flip=args.tta_flip,
                zoom=args.zoom, zoom_mass=args.zoom_mass,
                balance_per_class=args.balance_per_class, balance_iters=args.balance_iters)


if __name__ == "__main__":
    main()
