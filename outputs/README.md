# outputs/

One folder per finished run, named after its experiment
(`configs/experiments/<EXPERIMENT>.yaml` → `outputs/<EXPERIMENT>/`). It holds the
small records that make a result checkable in a pull request. The large files
live on Google Drive.

## What goes where

| File | Git (`outputs/<EXPERIMENT>/`) | Team Drive (`CS604_RUNS/<EXPERIMENT>/`) |
| --- | :---: | :---: |
| `run.json`: commit, GPU, parameters, budget, accuracy | ✓ | ✓ |
| `config.json`: the resolved config | ✓ | ✓ |
| `metrics.csv`: loss and accuracy per step/epoch | ✓ | ✓ |
| `baseline.json`, `hardware.json`, `checkpoint_audit.json` | ✓ | ✓ |
| `checkpoint_hashes.jsonl`: SHA-256 of each checkpoint | ✓ | ✓ |
| `sync.json`, `leaderboard.json` (CodaBench ID and score) | ✓ | ✓ |
| `*.log`, `analysis/*.json` | ✓ | ✓ |
| `submissions/*.inference.json`: prediction audit and latency | ✓ | ✓ |
| `artifacts.json`: list of the large files below, with sizes | ✓ | |
| `best.pt`, `last.pt`, `baseline.pt` (35 MB–1.1 GB each) | ✗ | ✓ |
| `checkpoints/*_step_*.pt` (resume points) | ✗ | only while the run is in progress |
| `splits/*.txt` (3–25 MB, rebuilt from `split_seed`) | ✗ | ✓ |
| `submissions/*.txt`, `*.zip` (prediction files) | ✗ | ✓ |
| `class_prototypes.pt`, feature caches | ✗ | ✓ |
| Course images and archives | ✗ | ✗ (download from Hugging Face) |
| Tokens (`secrets/`) | ✗ never | your own Drive only, never the team folder |

`.gitignore` enforces the git column: inside `outputs/`, only the file names
ticked above can be committed. GitHub rejects files over 100 MB, and checkpoints
would fill a free account's storage quickly.

## Publishing a run

After a run has finished and is in `cs604-outputs/<EXPERIMENT>/` (for a Colab
run, after `scripts/sync_run.py`):

```bash
python scripts/publish_run.py <EXPERIMENT>
git add outputs/<EXPERIMENT>
git commit -m "Add <EXPERIMENT> results"
```

`publish_run.py` copies the record files, rewrites absolute paths to
`${CS604_OUTPUTS}`, `${CS604_DATA}` or `~`, and writes `artifacts.json`. Then copy
the whole run folder (without `checkpoints/`) to the team Drive folder
`CS604_RUNS/<EXPERIMENT>/` so teammates can load `best.pt` or resubmit.

To use someone else's checkpoint, download `CS604_RUNS/<EXPERIMENT>/` into your
`cs604-outputs/<EXPERIMENT>/` and check `best.pt` against the hash in
`checkpoint_hashes.jsonl`.
