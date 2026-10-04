from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import publish_run  # noqa: E402


class PublishRunTest(unittest.TestCase):
    def test_copies_records_and_lists_large_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "cs604-outputs"
            run = local / "e001"
            (run / "checkpoints").mkdir(parents=True)
            (run / "splits").mkdir()
            (run / "submissions").mkdir()
            (run / "run.json").write_text(
                json.dumps({"output_dir": (run).as_posix(), "status": "completed"}), encoding="utf-8"
            )
            (run / "metrics.csv").write_text("step,loss\n1,9.0\n", encoding="utf-8")
            (run / "best.pt").write_bytes(b"x" * 10)
            (run / "checkpoints" / "resume_step_000001.pt").write_bytes(b"x")
            (run / "splits" / "internal_val.txt").write_text("a 0\n", encoding="utf-8")
            (run / "submissions" / "val_team_08.txt").write_text("a 0\n", encoding="utf-8")
            (run / "submissions" / "val_team_08.inference.json").write_text("{}", encoding="utf-8")

            dest = Path(tmp) / "repo" / "outputs" / "e001"
            copied, left = publish_run.publish_run(run, dest, local)

            self.assertEqual(copied, ["metrics.csv", "run.json", "submissions/val_team_08.inference.json"])
            self.assertEqual(
                [item["path"] for item in left],
                ["best.pt", "splits/internal_val.txt", "submissions/val_team_08.txt"],
            )
            published = json.loads((dest / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(published["output_dir"], "${CS604_OUTPUTS}/e001")
            self.assertFalse((dest / "best.pt").exists())
            self.assertTrue((dest / "artifacts.json").is_file())

    def test_refuses_a_folder_that_is_not_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                publish_run.publish_run(Path(tmp), Path(tmp) / "out", Path(tmp))


if __name__ == "__main__":
    unittest.main()
