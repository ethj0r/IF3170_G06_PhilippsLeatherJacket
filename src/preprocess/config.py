"""Configuration constants and dataclass for the preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

ID_COLUMN = "id"
TARGET_COLUMN = "is_canceled"

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

ORDINAL_MAPS = {
    "meal_package": {
        "Room Only": 1,
        "Breakfast": 2,
        "Half Board": 3,
        "Full Board": 4,
    }
}

ONE_HOT_COLUMNS = ["room_category", "booking_channel"]

DEFAULT_RANDOM_STATE = 824
DEFAULT_TOP_K_FEATURES = 15
DEFAULT_PCA_VARIANCE = 0.95


@dataclass
class PreprocessingConfig:
    """Configuration options controlling each preprocessing stage."""

    numerical_columns: list[str] = field(default_factory=lambda: list(NUMERICAL_COLUMNS))
    categorical_columns: list[str] = field(default_factory=lambda: list(CATEGORICAL_COLUMNS))
    ordinal_maps: dict[str, dict[str, int]] = field(
        default_factory=lambda: {k: dict(v) for k, v in ORDINAL_MAPS.items()}
    )
    one_hot_columns: list[str] = field(default_factory=lambda: list(ONE_HOT_COLUMNS))
    id_column: str = ID_COLUMN
    target_column: str = TARGET_COLUMN
    random_state: int = DEFAULT_RANDOM_STATE
    top_k_features: int = DEFAULT_TOP_K_FEATURES
    pca_variance: float | int = DEFAULT_PCA_VARIANCE
    use_train_ref_for_test_imputation: bool = True
    use_scaling: bool = True
    use_smote: bool = True
    use_feature_selection: bool = True
    use_pca: bool = True
    verbose: bool = True
