"""Stage 2: Duplicate detection and invalid row handling."""

from __future__ import annotations

import pandas as pd


def check_duplicates(
    df: pd.DataFrame,
    name: str = "Train",
    verbose: bool = True,
) -> int:
    """Count duplicated rows in a DataFrame."""
    dup_count = int(df.duplicated().sum())
    if verbose:
        print(f"Nilai ganda {name.lower()} = {dup_count}\n")
    return dup_count


def detect_invalid_rows(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Identify rows with negative numbers, zero adults, zero nights, or invalid arrival dates."""
    numeric_df = df.select_dtypes(include="number")
    has_negative = (numeric_df < 0).any(axis=1)
    no_adults = df["num_adults"] == 0
    no_nights = (df["weekend_nights"] + df["weekday_nights"]) == 0

    date_df = df[["arrival_year", "arrival_month", "arrival_day"]].rename(
        columns={"arrival_year": "year", "arrival_month": "month", "arrival_day": "day"}
    )
    invalid_dates = pd.to_datetime(date_df, errors="coerce").isna()

    return {
        "has_negative": has_negative,
        "no_adults": no_adults,
        "no_nights": no_nights,
        "invalid_dates": invalid_dates,
    }


def _log_invalid_summary(masks: dict[str, pd.Series], numeric_neg_cells: int, split_name: str) -> None:
    """Print invalid-value counts for a dataset split."""
    print(f"Jumlah nilai negatif pada data {split_name}: {numeric_neg_cells}")
    print(f"Banyak pesanan tanpa orang dewasa ({split_name}) = {int(masks['no_adults'].sum())}")
    print(f"Banyak pesanan dengan jumlah malam 0 ({split_name}) = {int(masks['no_nights'].sum())}")
    print(f"Banyak tanggal invalid ({split_name}): {int(masks['invalid_dates'].sum())}")


def clean_invalid_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop invalid rows from training data and repair invalid values in test data using train medians."""
    train_cleaned = train_df.copy()
    train_masks = detect_invalid_rows(train_cleaned)
    if verbose:
        train_neg_cells = int((train_cleaned.select_dtypes(include="number") < 0).sum().sum())
        _log_invalid_summary(train_masks, train_neg_cells, "train")

    valid_mask = (
        (~train_masks["has_negative"])
        & (~train_masks["no_adults"])
        & (~train_masks["no_nights"])
        & (~train_masks["invalid_dates"])
    )
    train_cleaned = train_cleaned[valid_mask].reset_index(drop=True)
    if verbose:
        print(f"Panjang train_df_cleaned setelah hapus data invalid = {len(train_cleaned)}\n")

    test_cleaned = test_df.copy()
    test_masks = detect_invalid_rows(test_cleaned)
    if verbose:
        test_neg_cells = int((test_cleaned.select_dtypes(include="number") < 0).sum().sum())
        _log_invalid_summary(test_masks, test_neg_cells, "test")

    median_train_adults = train_cleaned["num_adults"].median()
    test_cleaned.loc[test_masks["no_adults"], "num_adults"] = median_train_adults

    median_weekday_nights = train_cleaned["weekday_nights"].median()
    test_cleaned.loc[test_masks["no_nights"], "weekday_nights"] = median_weekday_nights

    if verbose:
        print(f"Panjang test_df_cleaned setelah fill data invalid = {len(test_cleaned)}\n")

    return train_cleaned, test_cleaned
