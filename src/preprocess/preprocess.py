"""Orchestrator combining all modular preprocessing stages into a single pipeline."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.preprocess.balancing import apply_smote, split_features_target
    from src.preprocess.cleaning import check_duplicates, clean_invalid_data
    from src.preprocess.config import PreprocessingConfig
    from src.preprocess.encoding import encode_categorical_features
    from src.preprocess.feature_selection import select_features
    from src.preprocess.missing import handle_missing_values
    from src.preprocess.reduction import apply_pca
    from src.preprocess.scaling import scale_numerical_features
else:
    from .balancing import apply_smote, split_features_target
    from .cleaning import check_duplicates, clean_invalid_data
    from .config import PreprocessingConfig
    from .encoding import encode_categorical_features
    from .feature_selection import select_features
    from .missing import handle_missing_values
    from .reduction import apply_pca
    from .scaling import scale_numerical_features


@dataclass
class PreprocessingResult:
    """Outputs and fitted transformers produced by the preprocessing pipeline."""

    train_cleaned: pd.DataFrame
    test_cleaned: pd.DataFrame
    train_encoded: pd.DataFrame
    test_encoded: pd.DataFrame
    X_train_encoded: pd.DataFrame
    y_train_encoded: pd.Series
    X_train_balanced: pd.DataFrame
    y_train_balanced: pd.Series
    X_train_final: pd.DataFrame
    X_test_final: pd.DataFrame
    X_train_pca: np.ndarray | None
    X_test_pca: np.ndarray | None
    selected_features: list[str]
    feature_summary: pd.DataFrame | None
    cumulative_variance: np.ndarray | None
    scaler: StandardScaler | None
    selector: SelectKBest | None
    pca: PCA | None

    @property
    def X_train(self) -> np.ndarray:
        """Final training feature matrix ready for model fitting (PCA output if enabled)."""
        if self.X_train_pca is not None:
            return self.X_train_pca
        return self.X_train_final.to_numpy(dtype=float)

    @property
    def X_test(self) -> np.ndarray:
        """Final test feature matrix aligned with X_train."""
        if self.X_test_pca is not None:
            return self.X_test_pca
        return self.X_test_final.to_numpy(dtype=float)

    @property
    def y_train(self) -> np.ndarray:
        """Training target array corresponding to X_train."""
        return np.asarray(self.y_train_balanced)

    @property
    def is_orig(self) -> np.ndarray:
        """Boolean mask marking original (non-SMOTE) rows in X_train/y_train."""
        return np.arange(len(self.y_train_balanced)) < len(self.X_train_encoded)


def run_preprocessing(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Execute preprocessing stages sequentially and return intermediate and final outputs."""
    cfg = config or PreprocessingConfig()

    # 1. Missing Value Handling
    train_cleaned, test_cleaned = handle_missing_values(
        train_df,
        test_df,
        numerical_columns=cfg.numerical_columns,
        categorical_columns=cfg.categorical_columns,
        use_train_ref_for_test=cfg.use_train_ref_for_test_imputation,
        verbose=cfg.verbose,
    )

    # 2. Duplicate & Invalid Data Handling
    check_duplicates(train_df, name="training", verbose=cfg.verbose)
    train_cleaned, test_cleaned = clean_invalid_data(
        train_cleaned,
        test_cleaned,
        verbose=cfg.verbose,
    )

    # 3. Categorical Encoding
    train_encoded, test_encoded = encode_categorical_features(
        train_cleaned,
        test_cleaned,
        ordinal_maps=cfg.ordinal_maps,
        one_hot_columns=cfg.one_hot_columns,
        target_column=cfg.target_column,
        verbose=cfg.verbose,
    )

    # 4. Numerical Scaling (optional via config.use_scaling)
    if cfg.use_scaling:
        train_encoded, test_encoded, scaler = scale_numerical_features(
            train_encoded,
            test_encoded,
            numerical_columns=cfg.numerical_columns,
            verbose=cfg.verbose,
        )
    else:
        scaler = None

    # 5. Split Features/Target & Handle Class Imbalance (optional via config.use_smote)
    X_train_encoded, y_train_encoded = split_features_target(
        train_encoded,
        target_column=cfg.target_column,
        id_column=cfg.id_column,
    )
    if cfg.use_smote:
        X_train_balanced, y_train_balanced = apply_smote(
            X_train_encoded,
            y_train_encoded,
            random_state=cfg.random_state,
            verbose=cfg.verbose,
        )
    else:
        X_train_balanced = X_train_encoded.copy()
        y_train_balanced = y_train_encoded.copy()

    # 6. Feature Selection (optional via config.use_feature_selection)
    if cfg.use_feature_selection:
        X_train_final, X_test_final, selected_features, feature_summary, selector = select_features(
            X_train_balanced,
            y_train_balanced,
            test_encoded,
            top_k=cfg.top_k_features,
            verbose=cfg.verbose,
        )
    else:
        selected_features = list(X_train_balanced.columns)
        X_train_final = X_train_balanced.copy()
        X_test_final = test_encoded[selected_features].copy()
        feature_summary = None
        selector = None

    # 7. Dimensionality Reduction via PCA (optional via config.use_pca)
    if cfg.use_pca:
        X_train_pca, X_test_pca, pca_final, cumulative_variance = apply_pca(
            X_train_final,
            X_test_final,
            n_components=cfg.pca_variance,
            random_state=cfg.random_state,
            verbose=cfg.verbose,
        )
    else:
        X_train_pca = None
        X_test_pca = None
        pca_final = None
        cumulative_variance = None

    return PreprocessingResult(
        train_cleaned=train_cleaned,
        test_cleaned=test_cleaned,
        train_encoded=train_encoded,
        test_encoded=test_encoded,
        X_train_encoded=X_train_encoded,
        y_train_encoded=y_train_encoded,
        X_train_balanced=X_train_balanced,
        y_train_balanced=y_train_balanced,
        X_train_final=X_train_final,
        X_test_final=X_test_final,
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca,
        selected_features=selected_features,
        feature_summary=feature_summary,
        cumulative_variance=cumulative_variance,
        scaler=scaler,
        selector=selector,
        pca=pca_final,
    )


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    raw_train = pd.read_csv(data_dir / "train.csv")
    raw_test = pd.read_csv(data_dir / "test.csv")
    result = run_preprocessing(raw_train, raw_test)
    print(f"Final shapes -> X_train: {result.X_train.shape}, X_test: {result.X_test.shape}")
