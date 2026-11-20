"""Stage 4: Standardization scaling for numerical features."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import NUMERICAL_COLUMNS


def scale_numerical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numerical_columns: list[str] = NUMERICAL_COLUMNS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on training numerical features and transform both train and test sets."""
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    scaler = StandardScaler()
    train_scaled[numerical_columns] = scaler.fit_transform(train_scaled[numerical_columns])
    test_scaled[numerical_columns] = scaler.transform(test_scaled[numerical_columns])

    if verbose:
        print(f"StandardScaler diterapkan pada {len(numerical_columns)} fitur numerik.\n")

    return train_scaled, test_scaled, scaler
