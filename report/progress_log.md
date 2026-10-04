# Progress log

One short note per run. After CodaBench, also add a row to `experiment_index.csv`.

## Template

```
### YYYY-MM-DD — name — config
- run_dir:
- val acc (ours / CodaBench):
- trainable % / peak VRAM / GPU-hours:
- what changed:
- next:
```

## Log

### 2026-10-02 — Van — prototype head (from `van` branch)
- class-mean features, no training → ~43.9% (matches the PhD zero-shot line)
- random head → ~0.28%
- ported into this trainer as `scripts/build_prototypes.py` + `head_init: prototype`
