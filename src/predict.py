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
def run_predict(cfg, ckpt_path, split, team, out_dir):
    device = get_device()
    _, eval_tf = build_transforms(cfg)
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

    with out_path.open("w", encoding="utf-8") as f:
        for images, _, paths in tqdm(loader, desc=f"predict {split}"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            for rel_path, pred in zip(paths, preds):
                f.write(f"{rel_path} {pred}\n")

    print(f"[predict] wrote {out_path} ({len(dataset)} lines)")
    return out_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, data_root=args.data_root, batch_size=args.batch_size)
    run_predict(cfg, args.ckpt, args.split, args.team, args.out_dir)


if __name__ == "__main__":
    main()
