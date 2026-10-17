"""Choose which pretrained weights to train, for the arms in configs/ablation_param_selection.yaml.

angle     Partial fine-tuning (Ye et al., 2023): rank layers by the fine-tuned angle
          arccos(<W_pre, W_ft> / (|W_pre| |W_ft|)) between the pretrained weights and a
          full fine-tuning run. For hard datasets the paper tunes the LARGEST angles.
    python scripts/select_params.py angle --config configs/lora_r24_336_recipe.yaml \\
        --ckpt <full FT run>/last.pt --out_dir <folder>
  writes angles.csv (every matrix, for the report), selected_layer.yaml (the paper's
  rule: whole ATTN / FFN layers, top angle first, within --layer_budget) and
  selected_matrix.yaml (finer than the paper: single matrices within --matrix_budget).

gradient  GPS (Zhang et al., CVPR 2024): for every output neuron of every Linear in the
          blocks (and the patch embedding), keep the top-K input connections by the
          magnitude of the loss gradient over --max_batches training batches.
    python scripts/select_params.py gradient --config configs/lora_r24_336_recipe.yaml \\
        --data_root /content/pilot_data/c1000_h5_s0 --prototype_file <pilot prototypes> \\
        --k 42 10 --out_dir <folder>
  writes gps_mask_k<K>.pt for each K. GPS scores with a supervised contrastive loss
  because its head starts random; ours starts from class prototypes, so plain
  cross-entropy through that head is used instead.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ImageListDataset, build_transforms
from src.heads import load_prototypes
from src.model import build_model
from src.utils import apply_overrides, get_device, load_config, set_seed

MATRICES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def scored_weights(model):
    """Backbone weights a selection may pick: patch embedding + every block Linear."""
    names = ["patch_embed.proj"] + [f"blocks.{i}.{m}" for i in range(len(model.blocks)) for m in MATRICES]
    return {name: model.get_submodule(name).weight for name in names}


def angle_deg(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm())
    return math.degrees(math.acos(float(cos.clamp(-1.0, 1.0))))


def cmd_angle(args):
    cfg = load_config(args.config)
    apply_overrides(cfg, method="linear_probe", head_init="random", pretrained=True)
    model = build_model(cfg)
    pre = {k: w.detach() for k, w in scored_weights(model).items()}
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if ckpt.get("trainable_only"):
        raise SystemExit(f"{args.ckpt} is a PEFT checkpoint; the angle needs a full fine-tuning run")

    rows = []
    for name, w in pre.items():
        rows.append({"module": name, "block": name.split(".")[1] if name.startswith("blocks.") else "embed",
                     "params": w.numel(), "angle_deg": angle_deg(w, state[f"{name}.weight"])})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "angles.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # The paper's unit: an ATTN layer (qkv + proj) or an FFN layer (fc1 + fc2), angle
    # over the layer's weights together, ranked within each group, top angles first.
    by_name = {r["module"]: r for r in rows}
    layers = []
    for i in range(len(model.blocks)):
        for kind, parts in (("attn", ("attn.qkv", "attn.proj")), ("ffn", ("mlp.fc1", "mlp.fc2"))):
            names = [f"blocks.{i}.{p}" for p in parts]
            a = torch.cat([pre[n].flatten() for n in names])
            b = torch.cat([state[f"{n}.weight"].flatten() for n in names])
            layers.append({"names": names, "kind": kind, "block": i,
                           "params": sum(by_name[n]["params"] for n in names), "angle": angle_deg(a, b)})
    layer_pick = greedy(sorted(layers, key=lambda r: -r["angle"]), args.layer_budget)
    matrix_pick = greedy(sorted(({"names": [r["module"]], **r} for r in rows), key=lambda r: -r["angle_deg"]),
                         args.matrix_budget)

    for title, pick, file in (("layer", layer_pick, "selected_layer.yaml"),
                              ("matrix", matrix_pick, "selected_matrix.yaml")):
        names = [n for r in pick for n in r["names"]]
        (out / file).write_text(yaml.safe_dump(names))
        print(f"[angle] {title}: {sum(r['params'] for r in pick):,} params -> {out / file}")
        for r in pick:
            print(f"    {', '.join(r['names'])}  {r.get('angle', r.get('angle_deg')):.3f} deg")
    print("[angle] top 5 layers:", ", ".join(f"{r['kind']}{r['block']} {r['angle']:.3f}"
                                              for r in sorted(layers, key=lambda r: -r["angle"])[:5]))


def greedy(ranked, budget):
    """Take items in rank order while they fit in the parameter budget."""
    pick, used = [], 0
    for r in ranked:
        if used + r["params"] <= budget:
            pick.append(r)
            used += r["params"]
    return pick


def cmd_gradient(args):
    cfg = load_config(args.config)
    apply_overrides(cfg, method="linear_probe", head_init="prototype", prototype_file=args.prototype_file,
                    pretrained=True, data_root=args.data_root)
    set_seed(int(cfg.seed))
    device = get_device()
    model = build_model(cfg).to(device)
    load_prototypes(cfg, model)
    model.eval()  # no dropout / drop path while scoring
    for p in model.parameters():
        p.requires_grad = False
    weights = scored_weights(model)
    for w in weights.values():
        w.requires_grad = True

    _, eval_tf = build_transforms(cfg)
    ds = ImageListDataset(args.data_root, "train", transform=eval_tf)
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers),
                        pin_memory=device.type == "cuda")
    use_bf16 = bool(cfg.bf16) and device.type == "cuda"
    seen = 0
    for i, (images, labels, _) in enumerate(loader):
        if i >= args.max_batches:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            loss = F.cross_entropy(model(images), labels, reduction="sum")
        loss.backward()  # gradients add up over batches: the gradient of the summed loss
        seen += labels.size(0)
    print(f"[gradient] scored on {seen} images ({min(args.max_batches, len(loader))} batches)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k in args.k:
        masks, total = {}, 0
        for name, w in weights.items():
            score = w.grad.abs().reshape(w.shape[0], -1)  # one row per output neuron
            top = score.topk(min(k, score.shape[1]), dim=1).indices
            mask = torch.zeros_like(score, dtype=torch.bool).scatter_(1, top, True).reshape(w.shape)
            masks[f"{name}.weight"] = mask.cpu()
            total += int(mask.sum())
        path = out / f"gps_mask_k{k}.pt"
        torch.save(masks, path)
        print(f"[gradient] K={k}: {total:,} entries in {len(masks)} weights -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("angle")
    a.add_argument("--config", required=True)
    a.add_argument("--ckpt", required=True, help="full fine-tuning checkpoint (not trainable-only)")
    a.add_argument("--out_dir", required=True)
    a.add_argument("--layer_budget", type=float, default=2.4e6, help="params; one ATTN layer is 2.36M")
    a.add_argument("--matrix_budget", type=float, default=3.54e6, help="params; = LoRA r24's share of the recipe")
    g = sub.add_parser("gradient")
    g.add_argument("--config", required=True)
    g.add_argument("--data_root", required=True)
    g.add_argument("--prototype_file", required=True)
    g.add_argument("--k", type=int, nargs="+", default=[42, 10])
    g.add_argument("--max_batches", type=int, default=100)
    g.add_argument("--out_dir", required=True)
    args = p.parse_args()
    {"angle": cmd_angle, "gradient": cmd_gradient}[args.cmd](args)


if __name__ == "__main__":
    main()
