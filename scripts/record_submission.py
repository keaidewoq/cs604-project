"""Record a CodaBench submission in its run folder's leaderboard.json.

The uploaded file (``.txt`` or ``.zip`` from predict.py) is matched by SHA-256
against the ``.inference.json`` written beside it, and that file's checkpoint
hash must match a checkpoint in the run folder. So every recorded submission
points to the exact checkpoint that produced it.

Run it again with the same --submission-id to add or correct the score later.
summarize_experiments.py reads leaderboard.json into the experiment index.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cs604.engine.checkpoint_io import sha256_file  # noqa: E402

DEFAULT_LOCAL = ROOT.parent / "cs604-outputs"


def inference_record(submission: Path) -> dict:
    path = submission.with_suffix(".inference.json")
    if not path.is_file():
        raise SystemExit(f"No {path.name} beside {submission}; generate the file with scripts/predict.py.")
    return json.loads(path.read_text(encoding="utf-8"))


def check_provenance(run_dir: Path, submission: Path) -> dict:
    """Return the inference record after checking file and checkpoint hashes."""

    record = inference_record(submission)
    digest = sha256_file(submission)
    if digest not in (record.get("submission_sha256"), record.get("zip_sha256")):
        raise SystemExit(
            f"{submission.name} does not match its .inference.json hashes; it was changed or "
            "predates hash logging. Regenerate it with scripts/predict.py."
        )
    checkpoints = {sha256_file(path): path.name for path in sorted(run_dir.glob("*.pt"))}
    checkpoint = checkpoints.get(record.get("checkpoint_sha256"))
    if checkpoint is None:
        raise SystemExit(
            f"The checkpoint behind {submission.name} is not in {run_dir}; "
            "predict from that run's best.pt/last.pt."
        )
    return {**record, "uploaded_sha256": digest, "checkpoint_name": checkpoint}


def record_submission(
    run_dir: Path,
    submission: Path,
    submission_id: str,
    score: float | None = None,
    phase: str = "",
    notes: str = "",
) -> dict:
    provenance = check_provenance(run_dir, submission)
    path = run_dir / "leaderboard.json"
    board = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    submissions = board.setdefault("submissions", [])
    entry = next((item for item in submissions if item["submission_id"] == submission_id), None)
    if entry is None:
        entry = {"submission_id": submission_id, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        submissions.append(entry)
    entry.update(
        {
            "file": submission.name,
            "file_sha256": provenance["uploaded_sha256"],
            "split": provenance.get("split"),
            "checkpoint": provenance["checkpoint_name"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "checkpoint_step": provenance.get("checkpoint_step"),
            "predict_device": provenance.get("device"),
        }
    )
    if score is not None:
        entry["public_accuracy"] = score
    if phase:
        entry["phase"] = phase
    if notes:
        entry["notes"] = notes

    scored = [item for item in submissions if item.get("public_accuracy") is not None]
    if scored:
        best = max(scored, key=lambda item: item["public_accuracy"])
        board["public_accuracy"] = best["public_accuracy"]
        board["submission_id"] = best["submission_id"]
    board.setdefault("notes", "")
    path.write_text(json.dumps(board, indent=2), encoding="utf-8")
    return entry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run folder name under --local-root (= EXPERIMENT), or a path")
    parser.add_argument("--submission", type=Path, required=True, help="the .txt or .zip you uploaded")
    parser.add_argument("--submission-id", required=True, help="CodaBench submission ID")
    parser.add_argument("--score", type=float, help="public accuracy as a fraction, e.g. 0.5123")
    parser.add_argument("--phase", default="", help="CodaBench phase, e.g. val or test")
    parser.add_argument("--notes", default="")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL)
    args = parser.parse_args(argv)

    run_dir = Path(args.run) if Path(args.run).is_dir() else args.local_root / args.run
    if not (run_dir / "run.json").is_file():
        raise SystemExit(f"{run_dir} is not a run folder; sync it first with scripts/sync_run.py.")
    if args.score is not None and not 0.0 <= args.score <= 1.0:
        raise SystemExit("--score is a fraction between 0 and 1 (51.2% -> 0.512)")
    entry = record_submission(run_dir, args.submission, args.submission_id, args.score, args.phase, args.notes)
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
