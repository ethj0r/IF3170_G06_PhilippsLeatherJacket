"""Tests for submission checkpointing and best-score history."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.svm.checkpoints import register_submission


def _write_submission(path: Path, positives: int = 3) -> None:
    pd.DataFrame(
        {"id": [f"id{i}" for i in range(10)], "is_canceled": [1] * positives + [0] * (10 - positives)}
    ).to_csv(path, index=False)


class TestCheckpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.sub = self.dir / "submission.csv"
        self.model = self.dir / "model.pkl"
        self.model.write_bytes(b"dummy-model")
        _write_submission(self.sub)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_best_public_and_holdout_tracking(self) -> None:
        first = register_submission(
            self.sub,
            name="first",
            public_score=0.80,
            holdout_accuracy=0.85,
            model_path=self.model,
            checkpoint_dir=self.dir / "checkpoints",
        )
        self.assertTrue(first["new_public_best"])
        self.assertTrue(first["new_holdout_best"])
        self.assertTrue(Path(first["best_public_archive"]).exists())
        self.assertTrue(Path(first["best_holdout_archive"]).exists())
        self.assertTrue(Path(first["archived_model"]).exists())

        second = register_submission(
            self.sub,
            name="second",
            public_score=0.79,
            holdout_accuracy=0.86,
            model_path=self.model,
            checkpoint_dir=self.dir / "checkpoints",
        )
        self.assertFalse(second["new_public_best"])
        self.assertTrue(second["new_holdout_best"])

        third = register_submission(
            self.sub,
            name="third",
            public_score=0.81,
            holdout_accuracy=0.84,
            model_path=self.model,
            checkpoint_dir=self.dir / "checkpoints",
        )
        self.assertTrue(third["new_public_best"])
        self.assertFalse(third["new_holdout_best"])

        ckpt = self.dir / "checkpoints"
        history = json.loads((ckpt / "history.json").read_text())
        self.assertEqual(len(history["records"]), 3)
        self.assertAlmostEqual(history["best_public"]["public_score"], 0.81)
        self.assertAlmostEqual(history["best_holdout"]["holdout_accuracy"], 0.86)
        self.assertTrue((ckpt / "best_public.json").exists())
        self.assertTrue((ckpt / "best_holdout.json").exists())
        self.assertEqual(len(list((ckpt / "submissions").glob("*.csv"))), 3)

    def test_tie_is_not_a_new_best(self) -> None:
        register_submission(
            self.sub,
            name="first",
            public_score=0.80,
            holdout_accuracy=0.85,
            checkpoint_dir=self.dir / "checkpoints",
        )
        tie = register_submission(
            self.sub,
            name="tie",
            public_score=0.80,
            holdout_accuracy=0.85,
            checkpoint_dir=self.dir / "checkpoints",
        )
        self.assertFalse(tie["new_public_best"])
        self.assertFalse(tie["new_holdout_best"])


if __name__ == "__main__":
    unittest.main()
