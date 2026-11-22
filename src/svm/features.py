"""Feature engineering for the hotel dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 824

NUMERICAL_COLUMNS = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "days_before_arrival",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
]

CATEGORICAL_COLUMNS = [
    "needs_parking",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "is_returning_guest",
    "meal_package",
    "room_category",
    "booking_channel",
]

MISSING_INDICATOR_COLUMNS = [
    "special_requests",
    "nightly_rate_idr",
    "days_before_arrival",
    "num_children",
    "num_adults",
    "meal_package",
    "needs_parking",
    "room_category",
]


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing-value indicators."""
    df = df.copy()
    for col in MISSING_INDICATOR_COLUMNS:
        df[f"miss_{col}"] = df[col].isna().astype(np.int8)
    return df


def fill_missing_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERICAL_COLUMNS:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def fill_missing_test(df: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERICAL_COLUMNS:
        df[col] = df[col].fillna(train_ref[col].median())
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna(train_ref[col].mode()[0])
    return df


def _invalid_mask(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include="number")
    no_adults = df["num_adults"] == 0
    no_nights = (df["weekend_nights"] + df["weekday_nights"]) == 0
    dates = df[["arrival_year", "arrival_month", "arrival_day"]].rename(
        columns={"arrival_year": "year", "arrival_month": "month", "arrival_day": "day"}
    )
    invalid_dates = pd.to_datetime(dates, errors="coerce").isna()
    return (numeric < 0).any(axis=1) | no_adults | no_nights | invalid_dates


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    valid = ~_invalid_mask(df)
    return df[valid].reset_index(drop=True)


def clean_test(df: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    no_adults = df["num_adults"] == 0
    no_nights = (df["weekend_nights"] + df["weekday_nights"]) == 0
    df["invalid_num_adults"] = no_adults.astype(np.int8)
    df["invalid_no_nights"] = no_nights.astype(np.int8)
    df.loc[no_adults, "num_adults"] = train_ref["num_adults"].median()
    df.loc[no_nights, "weekday_nights"] = train_ref["weekday_nights"].median()
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dates = pd.to_datetime(
        df[["arrival_year", "arrival_month", "arrival_day"]].rename(
            columns={
                "arrival_year": "year",
                "arrival_month": "month",
                "arrival_day": "day",
            }
        )
    )
    df["dow"] = dates.dt.dayofweek.astype(int)
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["quarter"] = dates.dt.quarter.astype(int)
    df["day_of_year"] = dates.dt.dayofyear.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    df["month_sin"] = np.sin(2 * np.pi * df["arrival_month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["arrival_month"] / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)

    df["total_nights"] = df["weekend_nights"] + df["weekday_nights"]
    df["total_guests"] = df["num_adults"] + df["num_children"]
    df["adults_children_ratio"] = df["num_adults"] / (df["num_children"] + 1.0)
    df["is_family"] = (df["num_children"] > 0).astype(np.int8)
    df["log_rate"] = np.log1p(df["nightly_rate_idr"].clip(lower=0))
    df["log_days"] = np.log1p(df["days_before_arrival"].clip(lower=0))
    nights = df["total_nights"].replace(0, np.nan)
    guests = df["total_guests"].replace(0, np.nan)
    df["price_per_night"] = df["nightly_rate_idr"] / nights
    df["price_per_person"] = df["nightly_rate_idr"] / guests
    df["price_per_night_person"] = df["nightly_rate_idr"] / (nights * guests)

    df["prior_total"] = df["prior_cancellations"] + df["prior_completed_bookings"]
    df["prior_cancel_ratio"] = df["prior_cancellations"] / (df["prior_total"] + 1.0)
    df["any_prior_cancel"] = (df["prior_cancellations"] > 0).astype(np.int8)
    df["any_prior_completed"] = (df["prior_completed_bookings"] > 0).astype(np.int8)

    df["is_last_minute"] = (df["days_before_arrival"] <= 7).astype(np.int8)
    df["is_long_lead"] = (df["days_before_arrival"] > 180).astype(np.int8)
    df["has_special_requests"] = (df["special_requests"] > 0).astype(np.int8)
    df["special_requests_log"] = np.log1p(df["special_requests"].clip(lower=0))
    df["is_high_month"] = df["arrival_month"].isin([6, 7, 8]).astype(np.int8)
    df["is_low_month"] = df["arrival_month"].isin([1, 12]).astype(np.int8)
    return df


def encode_categoricals(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    meal_map = {
        "Room Only": 1,
        "Breakfast": 2,
        "Half Board": 3,
        "Full Board": 4,
    }
    train["meal_package_ord"] = train["meal_package"].map(meal_map)
    test["meal_package_ord"] = test["meal_package"].map(meal_map)

    raw_map = {
        "meal_package": "meal_package_raw",
        "room_category": "room_category_raw",
        "booking_channel": "booking_channel_raw",
        "arrival_month": "arrival_month_raw",
        "dow": "dow_raw",
    }
    for col, raw_col in raw_map.items():
        train[raw_col] = train[col]
        test[raw_col] = test[col]

    one_hot_columns = [
        "meal_package",
        "room_category",
        "booking_channel",
        "arrival_month",
        "dow",
    ]
    train = pd.get_dummies(train, columns=one_hot_columns, drop_first=True, dtype=np.float64)
    test = pd.get_dummies(test, columns=one_hot_columns, drop_first=True, dtype=np.float64)

    for col in train.columns:
        if col not in test.columns:
            test[col] = 0.0
    test = test[train.columns]
    return train, test


def prepare_engineered_frames(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return engineered train and test frames."""
    train = add_missing_indicators(train_raw)
    test = add_missing_indicators(test_raw)

    train = fill_missing_train(train)
    test = fill_missing_test(test, train)

    train = clean_train(train)
    test = clean_test(test, train)

    train = add_engineered_features(train)
    test = add_engineered_features(test)

    return encode_categoricals(train, test)


TARGET_ENCODING_KEYS = {
    "te_channel": ["booking_channel_raw"],
    "te_room": ["room_category_raw"],
    "te_meal": ["meal_package_raw"],
    "te_channel_room": ["booking_channel_raw", "room_category_raw"],
    "te_month_channel": ["arrival_month_raw", "booking_channel_raw"],
    "te_room_meal": ["room_category_raw", "meal_package_raw"],
    "te_year_month": ["arrival_year", "arrival_month_raw"],
    "te_dow_channel": ["dow_raw", "booking_channel_raw"],
}


def add_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add smoothed target encoding."""
    train = train.copy()
    test = test.copy()
    prior = float(train["is_canceled"].mean())

    for name, cols in TARGET_ENCODING_KEYS.items():
        key_train = train[cols].astype(str).agg("|".join, axis=1)
        key_test = test[cols].astype(str).agg("|".join, axis=1)
        agg = (
            pd.DataFrame({"k": key_train, "y": train["is_canceled"]})
            .groupby("k")["y"]
            .agg(["sum", "count"])
        )
        encoded = (agg["sum"] + prior * alpha) / (agg["count"] + alpha)
        train[name] = key_train.map(encoded).fillna(prior).to_numpy(float)
        test[name] = key_test.map(encoded).fillna(prior).to_numpy(float)

    return train, test


def robust_scale_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a median/IQR scaler."""
    X = np.asarray(X, dtype=np.float64)
    median = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    scale = np.where(iqr > 0, iqr, 1.0)
    return median, scale


def robust_scale_transform(
    X: np.ndarray, median: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (np.asarray(X, dtype=np.float64) - median) / scale


def standard_scale_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean, std


def standard_scale_transform(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (np.asarray(X, dtype=np.float64) - mean) / std


def stratified_split_indices(
    y: np.ndarray,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stratified train/validation indices."""
    y = np.asarray(y).ravel()
    rng = np.random.RandomState(random_state)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        n_val = int(round(test_size * len(indices)))
        n_val = max(1, min(len(indices) - 1, n_val))
        val_parts.append(indices[:n_val])
        train_parts.append(indices[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _stratified_folds(y: np.ndarray, n_splits: int, random_state: int) -> list[np.ndarray]:
    y = np.asarray(y).ravel()
    rng = np.random.RandomState(random_state)
    folds: list[list[np.ndarray]] = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for fold, part in enumerate(np.array_split(indices, n_splits)):
            folds[fold].append(part)
    return [np.concatenate(parts) for parts in folds]


def _select_distance_features(
    X: np.ndarray, y: np.ndarray, n_features: int
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    x_std = X_centered.std(axis=0)
    y_std = y_centered.std()
    denom = x_std * (y_std if y_std > 0 else 1.0)
    denom = np.where(denom > 0, denom, 1.0)
    corr = (X_centered * y_centered[:, None]).mean(axis=0) / denom
    order = np.argsort(-np.abs(corr))
    return order[: min(n_features, X.shape[1])]


def _knn_feature_block(
    query: np.ndarray,
    reference: np.ndarray,
    y_reference: np.ndarray,
    feature_idx: np.ndarray,
    k_values: tuple[int, ...] = (5, 15, 50),
    weighted_k: int = 50,
    chunk_size: int = 512,
) -> np.ndarray:
    """Compute KNN target and distance features."""
    Q = np.asarray(query, dtype=np.float64)[:, feature_idx]
    R = np.asarray(reference, dtype=np.float64)[:, feature_idx]
    y_ref = np.asarray(y_reference, dtype=np.float64).ravel()
    n_query = Q.shape[0]
    n_ref = R.shape[0]
    k_max = max(max(k_values), weighted_k)
    k_max = min(k_max, n_ref)
    n_target_features = len(k_values) + 1 + 4  # target means, weighted mean, distances
    out = np.empty((n_query, n_target_features), dtype=np.float64)

    r_norm = np.einsum("ij,ij->i", R, R)
    pos_mask = y_ref > 0.5
    neg_mask = ~pos_mask
    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        q = Q[start:end]
        q_norm = np.einsum("ij,ij->i", q, q)
        distances = q_norm[:, None] + r_norm[None, :] - 2.0 * (q @ R.T)
        np.maximum(distances, 0.0, out=distances)

        kth = np.argpartition(distances, k_max - 1, axis=1)[:, :k_max]
        kth_dist = np.take_along_axis(distances, kth, axis=1)
        order = np.argsort(kth_dist, axis=1)
        kth = np.take_along_axis(kth, order, axis=1)
        kth_dist = np.take_along_axis(kth_dist, order, axis=1)
        neighbor_y = y_ref[kth]

        columns = []
        for k in k_values:
            columns.append(neighbor_y[:, : min(k, n_ref)].mean(axis=1))
        weights = 1.0 / (kth_dist[:, : min(weighted_k, n_ref)] + 1e-6)
        weighted_mean = (
            neighbor_y[:, : min(weighted_k, n_ref)] * weights
        ).sum(axis=1) / weights.sum(axis=1)
        columns.append(weighted_mean)
        columns.append(distances[:, pos_mask].min(axis=1) if pos_mask.any() else np.full(end - start, np.nan))
        columns.append(distances[:, neg_mask].min(axis=1) if neg_mask.any() else np.full(end - start, np.nan))
        dist_pos = columns[-2]
        dist_neg = columns[-1]
        columns.append(dist_pos - dist_neg)
        columns.append(kth_dist[:, : min(15, n_ref)].mean(axis=1))
        out[start:end] = np.column_stack(columns)
    return out


def add_knn_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    *,
    n_distance_features: int = 25,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    scale_features: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]:
    """Add out-of-fold KNN features."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train).ravel()
    feature_idx = _select_distance_features(X_train, y_train, n_distance_features)

    folds = _stratified_folds(y_train, n_splits=n_splits, random_state=random_state)
    knn_train = None
    for fold in folds:
        mask = np.ones(len(y_train), dtype=bool)
        mask[fold] = False
        block = _knn_feature_block(
            X_train[fold], X_train[mask], y_train[mask], feature_idx
        )
        if knn_train is None:
            knn_train = np.empty((len(y_train), block.shape[1]), dtype=np.float64)
        knn_train[fold] = block

    knn_val = None
    if X_val is not None:
        knn_val = _knn_feature_block(
            np.asarray(X_val, dtype=np.float64), X_train, y_train, feature_idx
        )
    knn_test = None
    if X_test is not None:
        knn_test = _knn_feature_block(
            np.asarray(X_test, dtype=np.float64), X_train, y_train, feature_idx
        )

    if scale_features:
        median, scale = robust_scale_fit(knn_train)
        knn_train = robust_scale_transform(knn_train, median, scale)
        if knn_val is not None:
            knn_val = robust_scale_transform(knn_val, median, scale)
        if knn_test is not None:
            knn_test = robust_scale_transform(knn_test, median, scale)

    names = [
        "knn_target_5",
        "knn_target_15",
        "knn_target_50",
        "knn_weighted_50",
        "knn_dist_pos",
        "knn_dist_neg",
        "knn_dist_diff",
        "knn_mean_dist_15",
    ]
    X_train_aug = np.hstack([X_train, knn_train])
    X_val_aug = np.hstack([np.asarray(X_val, dtype=np.float64), knn_val]) if X_val is not None else None
    X_test_aug = np.hstack([np.asarray(X_test, dtype=np.float64), knn_test]) if X_test is not None else None
    return X_train_aug, X_val_aug, X_test_aug, names
