"""Tests for the engineered feature pipeline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import RobustScaler, StandardScaler

from src.svm.features import (
    add_target_encoding,
    prepare_engineered_frames,
    robust_scale_fit,
    robust_scale_transform,
    standard_scale_fit,
    standard_scale_transform,
    stratified_split_indices,
)


def _frame(n: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "id": [f"{i:08x}" for i in range(n)],
            "num_adults": rng.choice([1.0, 2.0, 3.0], n),
            "num_children": rng.choice([0.0, 1.0, np.nan], n),
            "weekend_nights": rng.randint(0, 3, n),
            "weekday_nights": rng.randint(1, 5, n),
            "meal_package": rng.choice(
                ["Room Only", "Breakfast", "Half Board", "Full Board"], n
            ),
            "needs_parking": rng.choice([0.0, 1.0, np.nan], n),
            "room_category": rng.choice(["Type A", "Type B", "Type C"], n),
            "days_before_arrival": rng.randint(0, 300, n).astype(float),
            "arrival_year": rng.choice([2023, 2024], n),
            "arrival_month": rng.randint(1, 13, n),
            "arrival_day": rng.randint(1, 29, n),
            "booking_channel": rng.choice(
                ["Online Travel Agent", "Offline Agent", "Corporate"], n
            ),
            "is_returning_guest": rng.randint(0, 2, n),
            "prior_cancellations": rng.randint(0, 3, n),
            "prior_completed_bookings": rng.randint(0, 3, n),
            "nightly_rate_idr": rng.uniform(0, 3_000_000, n),
            "special_requests": rng.randint(0, 5, n).astype(float),
            "is_canceled": rng.randint(0, 2, n),
        }
    )


class TestEngineeredFeatures(unittest.TestCase):
    def test_prepare_engineered_frames(self) -> None:
        train_raw = _frame(seed=1)
        test_raw = _frame(seed=2)
        train_raw.loc[0, "special_requests"] = np.nan
        train_raw.loc[1, "num_children"] = np.nan
        test_raw.loc[0, "nightly_rate_idr"] = np.nan

        train, test = prepare_engineered_frames(train_raw, test_raw)
        self.assertGreater(train.shape[1], train_raw.shape[1])
        self.assertGreater(test.shape[1], test_raw.shape[1])
        self.assertIn("miss_special_requests", train.columns)
        self.assertIn("te_channel", add_target_encoding(train, test)[0].columns)
        self.assertIn("total_nights", train.columns)
        self.assertIn("price_per_night", train.columns)
        self.assertIn("dow_raw", train.columns)
        self.assertIn("booking_channel_raw", train.columns)

        feature_cols = [
            c
            for c in train.columns
            if c not in ("id", "is_canceled") and not c.endswith("_raw")
        ]
        self.assertFalse(train[feature_cols].isna().any().any())
        self.assertFalse(test[feature_cols].isna().any().any())

    def test_target_encoding_unseen_category(self) -> None:
        train_raw = _frame(seed=3)
        test_raw = _frame(seed=4)
        train_raw.loc[0, "booking_channel"] = "Corporate"
        test_raw.loc[0, "booking_channel"] = "Airline"
        train, test = prepare_engineered_frames(train_raw, test_raw)
        train_te, test_te = add_target_encoding(train, test, alpha=10.0)

        te_cols = [c for c in train_te.columns if c.startswith("te_")]
        self.assertTrue(te_cols)
        self.assertFalse(train_te[te_cols].isna().any().any())
        self.assertFalse(test_te[te_cols].isna().any().any())
        # Unseen Airline should fall back to the training prior.
        prior = float(train_te["is_canceled"].mean())
        self.assertAlmostEqual(float(test_te.loc[0, "te_channel"]), prior, places=12)


class TestNumpyPreprocessing(unittest.TestCase):
    def test_robust_scaler_matches_sklearn(self) -> None:
        rng = np.random.RandomState(0)
        X = rng.normal(size=(100, 5))
        X[:, 0] *= 10
        X[0, 1] = 100.0

        loc, scale = robust_scale_fit(X)
        ours = robust_scale_transform(X, loc, scale)
        theirs = RobustScaler().fit_transform(X)
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)

    def test_standard_scaler_matches_sklearn(self) -> None:
        rng = np.random.RandomState(1)
        X = rng.normal(size=(100, 4)) * np.array([1.0, 2.0, 3.0, 4.0])
        mean, std = standard_scale_fit(X)
        ours = standard_scale_transform(X, mean, std)
        theirs = StandardScaler().fit_transform(X)
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)

    def test_stratified_split(self) -> None:
        y = np.array([0] * 70 + [1] * 30)
        train_idx, val_idx = stratified_split_indices(y, test_size=0.15, random_state=824)
        self.assertEqual(len(train_idx) + len(val_idx), len(y))
        self.assertEqual(len(set(train_idx) & set(val_idx)), 0)
        self.assertAlmostEqual(float(np.mean(y[val_idx])), 0.3, places=1)
        self.assertAlmostEqual(float(np.mean(y[train_idx])), 0.3, places=1)


if __name__ == "__main__":
    unittest.main()
