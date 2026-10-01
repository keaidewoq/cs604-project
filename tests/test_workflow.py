"""sync_run.py and record_submission.py on fake run folders (no GPU, no Drive)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sync_run = _load("sync_run")
record_submission = _load("record_submission")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SyncRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.drive = root / "drive" / "e001"
        self.local = root / "local" / "e001"
        (self.drive / "checkpoints").mkdir(parents=True)
        (self.drive / "splits").mkdir()
        (self.drive / "run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        (self.drive / "metrics.csv").write_text("step,loss\n1,2\n", encoding="utf-8")
        (self.drive / "splits" / "internal_val.txt").write_text("a 0\n", encoding="utf-8")
        (self.drive / "last.pt").write_bytes(b"last-v2")
        (self.drive / "checkpoints" / "resume_step_000001.pt").write_bytes(b"last-v1")
        (self.drive / "checkpoints" / "resume_step_000002.pt").write_bytes(b"last-v2")
        entries = [
            {"dest": "/content/drive/MyDrive/cs604/outputs/e001/last.pt", "sha256": _sha(b"last-v1")},
            {"dest": "/content/drive/MyDrive/cs604/outputs/e001/last.pt", "sha256": _sha(b"last-v2")},
        ]
        (self.drive / "checkpoint_hashes.jsonl").write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
        )

    def test_copies_everything_but_numbered_checkpoints_and_is_repeatable(self) -> None:
        verified = sync_run.sync_run(self.drive, self.local)
        self.assertIn("last.pt", verified)
        self.assertIn("splits/internal_val.txt", verified)
        self.assertFalse((self.local / "checkpoints").exists())
        self.assertEqual((self.local / "last.pt").read_bytes(), b"last-v2")
        self.assertTrue((self.local / "sync.json").is_file())
        sync_run.sync_run(self.drive, self.local)  # second run: same content, no error

    def test_half_synced_checkpoint_is_rejected(self) -> None:
        (self.drive / "last.pt").write_bytes(b"last-v2-partial")
        with self.assertRaisesRegex(SystemExit, "hash training logged"):
            sync_run.sync_run(self.drive, self.local)

    def test_running_run_is_refused_and_local_copy_never_overwritten(self) -> None:
        (self.drive / "run.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "not finished"):
            sync_run.sync_run(self.drive, self.local)
        (self.drive / "run.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        self.local.mkdir(parents=True)
        (self.local / "metrics.csv").write_text("different\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
            sync_run.sync_run(self.drive, self.local)

    def test_prune_removes_only_numbered_checkpoints(self) -> None:
        sync_run.sync_run(self.drive, self.local)
        removed = sync_run.prune_numbered_checkpoints(self.drive)
        self.assertEqual(removed, len(b"last-v1") + len(b"last-v2"))
        self.assertEqual(list((self.drive / "checkpoints").iterdir()), [])
        self.assertTrue((self.drive / "last.pt").is_file())


class RecordSubmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name) / "e001"
        (self.run / "submissions").mkdir(parents=True)
        (self.run / "run.json").write_text("{}", encoding="utf-8")
        (self.run / "best.pt").write_bytes(b"best")
        self.submission = self.run / "submissions" / "val_team_01.txt"
        self.submission.write_text("val/a.jpg 3\n", encoding="utf-8")
        self.inference = {
            "split": "val",
            "checkpoint_sha256": _sha(b"best"),
            "checkpoint_step": 12500,
            "device": "test",
            "submission_sha256": _sha(self.submission.read_bytes()),
        }
        self._write_inference()

    def _write_inference(self) -> None:
        self.submission.with_suffix(".inference.json").write_text(
            json.dumps(self.inference), encoding="utf-8"
        )

    def _board(self) -> dict:
        return json.loads((self.run / "leaderboard.json").read_text(encoding="utf-8"))

    def test_records_then_updates_score_and_tracks_best(self) -> None:
        record_submission.record_submission(self.run, self.submission, "111")
        self.assertNotIn("public_accuracy", self._board())
        record_submission.record_submission(self.run, self.submission, "111", score=0.51)
        record_submission.record_submission(self.run, self.submission, "222", score=0.49)
        board = self._board()
        self.assertEqual(len(board["submissions"]), 2)
        self.assertEqual((board["public_accuracy"], board["submission_id"]), (0.51, "111"))
        self.assertEqual(board["submissions"][0]["checkpoint"], "best.pt")

    def test_edited_file_is_rejected(self) -> None:
        self.submission.write_text("val/a.jpg 4\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "does not match"):
            record_submission.record_submission(self.run, self.submission, "111")

    def test_checkpoint_from_another_run_is_rejected(self) -> None:
        self.inference["checkpoint_sha256"] = _sha(b"other-run")
        self._write_inference()
        with self.assertRaisesRegex(SystemExit, "not in"):
            record_submission.record_submission(self.run, self.submission, "111")


if __name__ == "__main__":
    unittest.main()
