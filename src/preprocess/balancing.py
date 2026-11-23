"""Stage 5: Feature/target splitting and SMOTE oversampling for class balance."""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE

from .config import DEFAULT_RANDOM_STATE, ID_COLUMN, TARGET_COLUMN


def split_features_target(
    train_df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y from the training DataFrame."""
    drop_cols = [col for col in (target_column, id_column) if col in train_df.columns]
    X_train = train_df.drop(columns=drop_cols).copy()
    y_train = train_df[target_column].copy()
    return X_train, y_train


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Resample training data using SMOTE to balance the target classes."""
    if verbose:
        print(f"Data sebelum SMOTE: {y_train.value_counts().to_dict()}")

    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    if verbose:
        print(f"Data setelah SMOTE: {y_train_balanced.value_counts().to_dict()}\n")

    return X_train_balanced, y_train_balanced
