"""Modular preprocessing package for IF3170 Tubes 2."""

from .balancing import apply_smote, split_features_target
from .cleaning import check_duplicates, clean_invalid_data, detect_invalid_rows
from .config import (
    CATEGORICAL_COLUMNS,
    DEFAULT_PCA_VARIANCE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TOP_K_FEATURES,
    ID_COLUMN,
    NUMERICAL_COLUMNS,
    ONE_HOT_COLUMNS,
    ORDINAL_MAPS,
    TARGET_COLUMN,
    PreprocessingConfig,
)
from .encoding import encode_categorical_features
from .feature_selection import select_features
from .missing import handle_missing_values, impute_columns, summarize_missing_values
from .preprocess import PreprocessingResult, run_preprocessing
from .reduction import apply_pca
from .scaling import scale_numerical_features

__all__ = [
    "CATEGORICAL_COLUMNS",
    "DEFAULT_PCA_VARIANCE",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_TOP_K_FEATURES",
    "ID_COLUMN",
    "NUMERICAL_COLUMNS",
    "ONE_HOT_COLUMNS",
    "ORDINAL_MAPS",
    "PreprocessingConfig",
    "PreprocessingResult",
    "TARGET_COLUMN",
    "apply_pca",
    "apply_smote",
    "check_duplicates",
    "clean_invalid_data",
    "detect_invalid_rows",
    "encode_categorical_features",
    "handle_missing_values",
    "impute_columns",
    "run_preprocessing",
    "scale_numerical_features",
    "select_features",
    "split_features_target",
    "summarize_missing_values",
]
