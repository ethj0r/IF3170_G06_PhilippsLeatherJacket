"""Stage 1: Missing value inspection and median/mode imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def summarize_missing_values(
    df: pd.DataFrame,
    name: str = "Dataset",
    verbose: bool = True,
) -> dict[str, object]:
    """Summarize total and percentage of missing values in a DataFrame."""
    total_cells = int(np.prod(df.shape))
    missing_per_column = df.isnull().sum()
    total_missing = int(missing_per_column.sum())
    percent_missing = (total_missing / total_cells) * 100 if total_cells > 0 else 0.0

    if verbose:
        print(f"{name} Missing Values")
        print(
            f"Total missing values = {total_missing}. "
            f"Missing percentage = {percent_missing}%\n"
        )

    return {
        "total_cells": total_cells,
        "missing_per_column": missing_per_column,
        "total_missing": total_missing,
        "percent_missing": percent_missing,
    }


def impute_columns(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    ref_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fill missing values in specified columns using median or mode from ref_df."""
    if df is None:
        raise ValueError("df cannot be None")
    if columns is None:
        raise ValueError("columns cannot be None")
    if method not in ("median", "mode"):
        raise ValueError(f"Invalid imputation method '{method}'. Expected 'median' or 'mode'.")

    result = df.copy()
    ref = ref_df if ref_df is not None else result

    for col in columns:
        if method == "median":
            filler = ref[col].median()
        else:
            filler = ref[col].mode()[0]
        result[col] = result[col].fillna(filler)

    return result


def handle_missing_values(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numerical_columns: list[str] = NUMERICAL_COLUMNS,
    categorical_columns: list[str] = CATEGORICAL_COLUMNS,
    use_train_ref_for_test: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect missing values and impute numerical (median) and categorical (mode) columns."""
    if verbose:
        summarize_missing_values(train_df, name="Train DF", verbose=True)
        summarize_missing_values(test_df, name="Test DF", verbose=True)
        print(f"Training dataset sebelum drop missing vals: {len(train_df)}")
        print(f"Training dataset setelah drop missing vals: {len(train_df.dropna())}\n")

    train_cleaned = impute_columns(train_df, numerical_columns, "median")
    train_cleaned = impute_columns(train_cleaned, categorical_columns, "mode")

    test_ref = train_df if use_train_ref_for_test else test_df
    test_cleaned = impute_columns(test_df, numerical_columns, "median", ref_df=test_ref)
    test_cleaned = impute_columns(test_cleaned, categorical_columns, "mode", ref_df=test_ref)

    if verbose:
        print(f"Sisa missing train_df values setelah cleaning = {train_cleaned.isnull().sum().sum()}")
        print(f"Panjang train_df_cleaned setelah cleaning = {len(train_cleaned)}")
        print(f"Sisa missing test_df values setelah cleaning = {test_cleaned.isnull().sum().sum()}")
        print(f"Panjang test_df_cleaned setelah cleaning = {len(test_cleaned)}\n")

    return train_cleaned, test_cleaned
