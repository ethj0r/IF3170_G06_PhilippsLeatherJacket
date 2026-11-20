"""Stage 7: Dimensionality reduction via Principal Component Analysis (PCA)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import DEFAULT_PCA_VARIANCE, DEFAULT_RANDOM_STATE


def apply_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: float | int = DEFAULT_PCA_VARIANCE,
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, PCA, np.ndarray]:
    """Compute cumulative variance curve and transform train/test features with PCA."""
    pca_eval = PCA(random_state=random_state)
    pca_eval.fit(X_train)
    cumulative_variance = np.cumsum(pca_eval.explained_variance_ratio_)

    pca_final = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca_final.fit_transform(X_train)
    X_test_pca = pca_final.transform(X_test)

    if verbose:
        print(f"Dimensi sebelum PCA (hasil feature selection): {X_train.shape[1]}")
        print(f"Dimensi setelah PCA (mempertahankan {n_components} varians): {X_train_pca.shape[1]}\n")

    return X_train_pca, X_test_pca, pca_final, cumulative_variance
