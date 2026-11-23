"""Model persistence and submission CSV generation with format assertions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def save_and_verify_model(
    model: Any,
    model_path: str | Path,
    verbose: bool = True,
) -> Any:
    """Persist a trained model with joblib, verify reloading, and return the reloaded instance."""
    target_path = Path(model_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, target_path)
    if verbose:
        size_kb = os.path.getsize(target_path) / 1024
        print(f"Model berhasil disimpan ke '{target_path}' (Ukuran: {size_kb:.2f} KB)")

    loaded_model = joblib.load(target_path)
    if verbose:
        print(f"Model berhasil di-load kembali dari '{target_path}'!\n")

    return loaded_model


def generate_submission(
    model: Any,
    X_test: np.ndarray,
    test_ids: pd.Series | np.ndarray,
    sample_submission: pd.DataFrame,
    output_csv_path: str | Path,
    prob_threshold: float | None = None,
    random_state: int = 824,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run predictions on test features, validate Kaggle submission format, and save CSV.

    If prob_threshold is provided and model implements predict_proba, binary predictions
    are computed with the specified threshold.
    """
    np.random.seed(random_state)

    if verbose:
        print("Melakukan prediksi pada data test...")

    if prob_threshold is not None and hasattr(model, "predict_proba"):
        test_probs = model.predict_proba(X_test)[:, 1]
        test_predictions = (test_probs >= prob_threshold).astype(int)
    else:
        test_predictions = model.predict(X_test)
    submission = pd.DataFrame(
        {
            "id": test_ids,
            "is_canceled": np.asarray(test_predictions, dtype=int),
        }
    )

    assert len(submission) == len(sample_submission), "Error: Jumlah baris submisi tidak sesuai!"
    assert list(submission.columns) == list(sample_submission.columns), "Error: Nama kolom tidak sesuai!"
    assert int(submission["is_canceled"].isnull().sum()) == 0, "Error: Ada nilai kosong (NaN) pada prediksi!"

    csv_path = Path(output_csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(csv_path, index=False)

    if verbose:
        print(f"File submisi '{csv_path}' berhasil dibuat dan siap dikumpulkan!")
        print("\nDistribusi hasil prediksi:")
        print(submission["is_canceled"].value_counts())

    return submission
