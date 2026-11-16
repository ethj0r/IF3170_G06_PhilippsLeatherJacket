"""Hotel preprocessing and data loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

from .features import (
    add_target_encoding,
    prepare_engineered_frames,
    robust_scale_fit,
    robust_scale_transform,
    standard_scale_fit,
    standard_scale_transform,
    stratified_split_indices,
)

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


@dataclass
class HotelData:
    X_tr: np.ndarray
    y_tr: np.ndarray
    orig_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    orig_val: np.ndarray
    X_test: np.ndarray
    test_ids: np.ndarray
    feature_names: list[str]
    feature_mode: str = "pca"


def _fill_missing(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ref = df if reference is None else reference
    for col in columns:
        if method == "median":
            filler = ref[col].median()
        elif method == "mode":
            filler = ref[col].mode()[0]
        else:
            raise ValueError(f"Unknown missing-value method {method!r}")
        df[col] = df[col].fillna(filler)
    return df


def _clean_train(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()
    df = _fill_missing(df, NUMERICAL_COLUMNS, "median")
    df = _fill_missing(df, CATEGORICAL_COLUMNS, "mode")

    numeric = df.select_dtypes(include="number")
    no_adults = df["num_adults"] == 0
    no_nights = (df["weekend_nights"] + df["weekday_nights"]) == 0
    date_df = df[["arrival_year", "arrival_month", "arrival_day"]].rename(
        columns={"arrival_year": "year", "arrival_month": "month", "arrival_day": "day"}
    )
    invalid_dates = pd.to_datetime(date_df, errors="coerce").isna()

    valid = (numeric >= 0).all(axis=1) & (~no_adults) & (~no_nights) & (~invalid_dates)
    return df[valid].reset_index(drop=True)


def _clean_test(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.copy()
    # The notebook fills test missing values using the test set's own
    # median/mode, so we mirror that behavior here.
    df = _fill_missing(df, NUMERICAL_COLUMNS, "median")
    df = _fill_missing(df, CATEGORICAL_COLUMNS, "mode")

    no_adults = df["num_adults"] == 0
    no_nights = (df["weekend_nights"] + df["weekday_nights"]) == 0

    df.loc[no_adults, "num_adults"] = train_df["num_adults"].median()
    df.loc[no_nights, "weekday_nights"] = train_df["weekday_nights"].median()
    return df


def _encode(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    ordinal_map = {
        "meal_package": {
            "Room Only": 1,
            "Breakfast": 2,
            "Half Board": 3,
            "Full Board": 4,
        }
    }
    for col, mapping in ordinal_map.items():
        train[col] = train[col].map(mapping)
        test[col] = test[col].map(mapping)

    train = pd.get_dummies(
        train, columns=["room_category", "booking_channel"], drop_first=True
    )
    test = pd.get_dummies(
        test, columns=["room_category", "booking_channel"], drop_first=True
    )

    # Keep the test matrix aligned with the training columns.  With this
    # dataset the categories are identical, so this does not change results.
    for col in train.columns:
        if col not in test.columns:
            test[col] = 0
    test = test[train.columns]
    return train, test


def load_hotel_data(
    data_dir: str | Path = "data",
    random_state: int = RANDOM_STATE,
    test_size: float = 0.15,
    feature_mode: str = "pca",
) -> HotelData:
    """Load and prepare the hotel dataset."""
    feature_mode = feature_mode.lower()
    if feature_mode not in {"pca", "top15", "all", "engineered"}:
        raise ValueError(
            f"Unknown feature_mode {feature_mode!r}. "
            "Choose from 'pca', 'top15', 'all', or 'engineered'."
        )
    if feature_mode == "engineered":
        return load_engineered_data(
            data_dir=data_dir,
            random_state=random_state,
            test_size=test_size,
        )
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "imbalanced-learn is required for the hotel preprocessing pipeline. "
            "Install it with: python3 -m pip install imbalanced-learn"
        ) from exc

    data_dir = Path(data_dir)
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")

    train_clean = _clean_train(train_raw)
    test_clean = _clean_test(test_raw, train_clean)

    train_encoded, test_encoded = _encode(train_clean, test_clean)

    scaler = StandardScaler()
    train_encoded[NUMERICAL_COLUMNS] = scaler.fit_transform(
        train_encoded[NUMERICAL_COLUMNS]
    )
    test_encoded[NUMERICAL_COLUMNS] = scaler.transform(
        test_encoded[NUMERICAL_COLUMNS]
    )

    X_train_encoded = train_encoded.drop(columns=["is_canceled", "id"]).copy()
    y_train_encoded = train_encoded["is_canceled"].copy()

    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_encoded, y_train_encoded
    )

    selector = SelectKBest(score_func=f_classif, k=10)
    selector.fit(X_train_balanced, y_train_balanced)
    feature_summary = pd.DataFrame(
        {"Feature": X_train_balanced.columns, "Score": selector.scores_}
    ).sort_values(by="Score", ascending=False)
    selected_features = feature_summary.head(15)["Feature"].tolist()

    X_train_final = X_train_balanced[selected_features]
    X_test_final = test_encoded[selected_features]

    if feature_mode == "pca":
        pca = PCA(n_components=0.95, random_state=random_state)
        X_train_model = pca.fit_transform(X_train_final)
        X_test_model = pca.transform(X_test_final)
        feature_names = [f"PC{i + 1}" for i in range(X_train_model.shape[1])]
    elif feature_mode == "top15":
        X_train_model = X_train_final.to_numpy(dtype=np.float64)
        X_test_model = X_test_final.to_numpy(dtype=np.float64)
        feature_names = list(selected_features)
    else:
        X_train_model = X_train_balanced.to_numpy(dtype=np.float64)
        X_test_model = test_encoded[X_train_encoded.columns].to_numpy(
            dtype=np.float64
        )
        feature_names = list(X_train_encoded.columns)

    n_orig = len(X_train_encoded)
    is_orig = np.arange(len(y_train_balanced)) < n_orig
    y_balanced = np.asarray(y_train_balanced).astype(int)
    strat_key = y_balanced * 2 + is_orig.astype(int)

    indices = np.arange(len(y_balanced))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key,
    )

    return HotelData(
        X_tr=np.asarray(X_train_model[train_idx], dtype=np.float64),
        y_tr=y_balanced[train_idx].astype(int),
        orig_tr=is_orig[train_idx],
        X_val=np.asarray(X_train_model[val_idx], dtype=np.float64),
        y_val=y_balanced[val_idx].astype(int),
        orig_val=is_orig[val_idx],
        X_test=np.asarray(X_test_model, dtype=np.float64),
        test_ids=test_raw["id"].to_numpy(),
        feature_names=feature_names,
        feature_mode=feature_mode,
    )


def _make_scaler(name: str):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "quantile":
        return QuantileTransformer(
            n_quantiles=1000,
            output_distribution="normal",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown scaler {name!r}")


def _engineered_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in ("id", "is_canceled") and not c.endswith("_raw")
    ]


def _scale_engineered(scaler_name: str, X_train, X_other, X_third=None):
    """Scale engineered features with NumPy."""
    scaler_name = scaler_name.lower()
    if scaler_name == "robust":
        loc, scale = robust_scale_fit(X_train)
        out = [robust_scale_transform(X_train, loc, scale)]
        for X in (X_other, X_third):
            if X is not None:
                out.append(robust_scale_transform(X, loc, scale))
        return tuple(out)
    if scaler_name == "standard":
        loc, scale = standard_scale_fit(X_train)
        out = [standard_scale_transform(X_train, loc, scale)]
        for X in (X_other, X_third):
            if X is not None:
                out.append(standard_scale_transform(X, loc, scale))
        return tuple(out)
    raise ValueError(
        "Engineered mode supports only NumPy scalers 'robust' and 'standard'; "
        f"got {scaler_name!r}"
    )


def load_engineered_data(
    data_dir: str | Path = "data",
    random_state: int = RANDOM_STATE,
    test_size: float = 0.15,
    scaler_name: str = "robust",
) -> HotelData:
    """Load engineered data with a clean train/validation split."""
    data_dir = Path(data_dir)
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")
    train_df, test_df = prepare_engineered_frames(train_raw, test_raw)

    train_idx, val_idx = stratified_split_indices(
        train_df["is_canceled"].to_numpy(),
        test_size=test_size,
        random_state=random_state,
    )
    tr = train_df.iloc[train_idx].reset_index(drop=True)
    val = train_df.iloc[val_idx].reset_index(drop=True)

    tr_te, val_te = add_target_encoding(tr, val, alpha=20.0)
    tr_te, test_te = add_target_encoding(tr, test_df, alpha=20.0)
    features = _engineered_feature_columns(tr_te)

    Xtr = tr_te[features].to_numpy(dtype=np.float64)
    Xval = val_te[features].to_numpy(dtype=np.float64)
    Xtest = test_te[features].to_numpy(dtype=np.float64)

    Xtr, Xval, Xtest = _scale_engineered(scaler_name, Xtr, Xval, Xtest)

    ytr = tr_te["is_canceled"].to_numpy(dtype=int)
    yval = val_te["is_canceled"].to_numpy(dtype=int)
    return HotelData(
        X_tr=np.asarray(Xtr, dtype=np.float64),
        y_tr=ytr,
        orig_tr=np.ones(len(ytr), dtype=bool),
        X_val=np.asarray(Xval, dtype=np.float64),
        y_val=yval,
        orig_val=np.ones(len(yval), dtype=bool),
        X_test=np.asarray(Xtest, dtype=np.float64),
        test_ids=test_raw["id"].to_numpy(),
        feature_names=features,
        feature_mode="engineered",
    )


def load_engineered_final(
    data_dir: str | Path = "data",
    random_state: int = RANDOM_STATE,
    scaler_name: str = "robust",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build engineered full-train and test matrices."""
    data_dir = Path(data_dir)
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")
    train_df, test_df = prepare_engineered_frames(train_raw, test_raw)
    train_te, test_te = add_target_encoding(train_df, test_df, alpha=20.0)
    features = _engineered_feature_columns(train_te)
    X_train = train_te[features].to_numpy(dtype=np.float64)
    y_train = train_te["is_canceled"].to_numpy(dtype=int)
    X_test = test_te[features].to_numpy(dtype=np.float64)
    X_train, X_test = _scale_engineered(scaler_name, X_train, X_test)
    return (
        np.asarray(X_train, dtype=np.float64),
        y_train,
        np.asarray(X_test, dtype=np.float64),
        test_raw["id"].to_numpy(),
        features,
    )
