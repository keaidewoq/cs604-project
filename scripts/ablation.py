"""Ablation at pilot scale: named arms, each one change from a base config.

python scripts/ablation.py --spec configs/ablation_lora_r24.yaml \\
    --data_root /content/data --output_dir /content/drive/MyDrive/CS604_RUNS

The spec file gives the base config, the pilot subset (as in pilot_sweep.py) and
the arms (config overrides). Steps:
  1. Build the pilot data root: a class subset with a labelled holdout.
  2. Rebuild the starting heads from the pilot's fit images only, so no arm's head
     has seen the holdout: prototypes (other classes keep `base_prototypes` rows)
     and, if an arm uses `head_init: probe`, a linear probe.
  3. Train each arm in its own process and score it on the holdout.
An arm may set `max_train_minutes: {match: <arm>}` to get exactly the training
minutes that arm used (it must run first), for comparisons at equal compute.
String values may contain `{out_dir}` (this spec's results folder), for files that
a step between runs writes there (scripts/select_params.py).
Writes <output_dir>/ablation_<spec>/ablation.csv and ablation_summary.json.
Rerunning with the same spec skips arms that already finished.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pilot_sweep import build_pilot_root, read_run_stats, summarise_run
from src.utils import Config, load_config, save_config

ENV = {**os.environ, "PYTHONPATH": str(ROOT)}


def run(cmd, log_path):
    """Run a subprocess, stream its output, keep a copy in log_path; return the output."""
    lines = []
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=ENV, cwd=ROOT)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        if proc.wait() != 0:
            raise SystemExit(f"failed (exit {proc.returncode}): {' '.join(map(str, cmd))}; log: {log_path}")
    return "".join(lines)


def clean_prototypes(base_cfg, pilot_root, out_dir, base_prototypes):
    out = out_dir / "prototypes_pilot_fit.pt"
    if out.is_file():
        print(f"[ablation] reusing {out}")
        return out
    cfg_path = out_dir / "heads_config.yaml"
    save_config(base_cfg, cfg_path)
    run([sys.executable, "-u", "scripts/build_prototypes.py", "--config", str(cfg_path),
         "--data_root", str(pilot_root), "--output", str(out),
         "--base_prototypes", str(base_prototypes)], out_dir / "prototypes.log")
    return out


def clean_probe(base_cfg, pilot_root, out_dir, feature_dir):
    done = out_dir / "probe_head.path"
    if done.is_file() and Path(done.read_text().strip()).is_file():
        print(f"[ablation] reusing {done.read_text().strip()}")
        return Path(done.read_text().strip())
    probe = load_config(ROOT / "configs" / "linear_probe_cached.yaml")
    for key in ("model_name", "pretrained", "num_classes", "img_size", "num_workers", "bf16", "seed"):
        if key in base_cfg:
            setattr(probe, key, base_cfg.get(key))
    cfg_path = out_dir / "probe_config.yaml"
    save_config(probe, cfg_path)
    text = run([sys.executable, "-u", "scripts/linear_probe_cached.py", "--config", str(cfg_path),
                "--team", "pilot", "--data_root", str(pilot_root), "--output_dir", str(out_dir),
                "--feature_dir", str(feature_dir)], out_dir / "probe.log")
    path = Path(re.findall(r"\[probe\] saved head: (\S+)", text)[-1])
    done.write_text(str(path))
    return path


def backbone_pct(text):
    m = re.findall(r"\[params\] backbone trainable \S+ \(([\d.]+)%", text)
    return float(m[-1]) if m else float("nan")


def resolve_matches(overrides, arms_dir):
    """Replace `{match: <arm>}` values with that arm's training minutes (from its marker),
    and `{out_dir}` in strings with the spec's results folder."""
    out = dict(overrides or {})
    for key, value in out.items():
        if isinstance(value, str) and "{out_dir}" in value:
            out[key] = value.replace("{out_dir}", str(arms_dir.parent))
            if not Path(out[key]).exists():
                raise SystemExit(f"{key}: {out[key]} does not exist yet (run scripts/select_params.py first)")
        if isinstance(value, dict) and "match" in value:
            marker = arms_dir / f"{value['match']}.json"
            if not marker.is_file():
                raise SystemExit(f"{key}: match {value['match']!r} has not run yet; list it earlier in the spec")
            out[key] = float(json.loads(marker.read_text())["training_min"])
            print(f"[ablation] {key} = {out[key]} (matched to {value['match']})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pilot_data_dir", default="/content/pilot_data")
    p.add_argument("--feature_dir", default="/content/features_pilot")
    p.add_argument("--only", nargs="*", help="Run only these arms")
    args = p.parse_args()

    spec_path = (ROOT / args.spec) if not Path(args.spec).is_absolute() else Path(args.spec)
    spec = yaml.safe_load(spec_path.read_text())
    base_path = spec_path.parent / spec["base"]
    base = load_config(base_path)
    base_d = {k: v for k, v in base.to_dict().items() if k != "pilot"}
    pilot = spec["pilot"]
    arms = spec["arms"]
    if args.only:
        arms = {k: v for k, v in arms.items() if k in args.only}

    pilot_root, info = build_pilot_root(args.data_root, args.pilot_data_dir,
                                        int(pilot["num_classes_subset"]), int(pilot["holdout_per_class"]),
                                        int(pilot.get("subset_seed", 0)))
    out_dir = Path(args.output_dir) / f"ablation_{spec_path.stem}"
    (out_dir / "arms").mkdir(parents=True, exist_ok=True)
    print(f"[ablation] folder {out_dir}  pilot data {info}")

    base_cfg = Config(base_d)
    needs_proto = any(str({**base_d, **o}.get("head_init")) == "prototype" for o in arms.values())
    needs_probe = any(str({**base_d, **o}.get("head_init")) == "probe" for o in arms.values())
    heads = {}
    if needs_proto:
        heads["prototype_file"] = str(clean_prototypes(base_cfg, pilot_root, out_dir, spec["base_prototypes"]))
    if needs_probe:
        heads["probe_head_file"] = str(clean_probe(base_cfg, pilot_root, out_dir, args.feature_dir))

    run_overrides = {k: v for k, v in pilot.items()
                     if k not in ("num_classes_subset", "holdout_per_class", "subset_seed")}
    results = []
    for name, overrides in arms.items():
        marker = out_dir / "arms" / f"{name}.json"
        if marker.is_file():
            print(f"[ablation] {name}: already done, reusing")
            results.append(json.loads(marker.read_text()))
            continue
        cfg = Config({**base_d, **run_overrides, **resolve_matches(overrides, out_dir / "arms"), **heads})
        cfg_path = out_dir / "arms" / f"{name}.yaml"
        save_config(cfg, cfg_path)
        print(f"\n[ablation] ===== {name}: {overrides or 'no change'}")
        text = run([sys.executable, "-u", "-m", "src.train", "--config", str(cfg_path),
                    "--data_root", str(pilot_root), "--output_dir", str(out_dir / "runs")],
                   out_dir / "arms" / f"{name}.log")
        run_dir = re.findall(r"\[train\] run_dir=(\S+)", text)[-1]
        row = summarise_run(run_dir, {"arm": name, "changes": json.dumps(overrides or {})})
        row["backbone_pct"] = backbone_pct(text)
        row["training_min"] = float(read_run_stats(run_dir).get("training_minutes", "nan"))
        marker.write_text(json.dumps(row))
        results.append(row)

    ref = next((r["final_val_acc"] for r in results if r["arm"].startswith("A_")), None)
    for r in results:
        r["delta_vs_A_points"] = round(100 * (r["final_val_acc"] - ref), 2) if ref is not None else None
    cols = ["arm", "changes", "final_val_acc", "delta_vs_A_points", "best_val_acc", "best_epoch",
            "final_val_loss", "early_dip", "backbone_pct", "peak_vram_gb", "training_min", "wall_min", "run_dir"]
    with (out_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    (out_dir / "ablation_summary.json").write_text(json.dumps(
        {"spec": str(spec_path), "base": str(base_path), "pilot_data": info, "heads": heads,
         "pilot_overrides": run_overrides, "results": results}, indent=2))

    print("\n[ablation] results (pilot holdout, final epoch)")
    for r in results:
        d = r["delta_vs_A_points"]
        print(f"  {r['arm']:18s} {r['final_val_acc']:.4f}  {'' if d is None else f'{d:+.2f} pts'}"
              f"  backbone {r['backbone_pct']}%{'  (early dip)' if r['early_dip'] else ''}")
    print(f"[ablation] wrote {out_dir / 'ablation.csv'}")


if __name__ == "__main__":
    main()
