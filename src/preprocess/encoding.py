"""Stage 3: Ordinal and one-hot encoding for categorical features."""

from __future__ import annotations

import pandas as pd

from .config import ONE_HOT_COLUMNS, ORDINAL_MAPS, TARGET_COLUMN


def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ordinal_maps: dict[str, dict[str, int]] = ORDINAL_MAPS,
    one_hot_columns: list[str] = ONE_HOT_COLUMNS,
    target_column: str = TARGET_COLUMN,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode ordinal and nominal categorical features and align test columns with train."""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col, mapping in ordinal_maps.items():
        train_encoded[col] = train_encoded[col].map(mapping)
        test_encoded[col] = test_encoded[col].map(mapping)

    train_encoded = pd.get_dummies(train_encoded, columns=one_hot_columns, drop_first=True)
    test_encoded = pd.get_dummies(test_encoded, columns=one_hot_columns, drop_first=True)

    # Align test feature columns to train columns (excluding target) so rare train-only categories exist in test.
    expected_test_columns = [col for col in train_encoded.columns if col != target_column]
    test_encoded = test_encoded.reindex(columns=expected_test_columns, fill_value=False)

    if verbose:
        print(
            f"Encoding selesai: train_encoded={train_encoded.shape}, "
            f"test_encoded={test_encoded.shape}\n"
        )

    return train_encoded, test_encoded
