"""K-Nearest Neighbors (KNN) classifier implemented from scratch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.models.base import BaseModel
    from src.preprocess.config import DEFAULT_RANDOM_STATE, PreprocessingConfig
else:
    from .base import BaseModel
    from ..preprocess.config import DEFAULT_RANDOM_STATE, PreprocessingConfig

SUPPORTED_METRICS = {"euclidean", "manhattan", "minkowski", "hamming"}
SUPPORTED_WEIGHTS = {"uniform", "inverted"}

DEFAULT_KNN_CONFIGS: list[dict[str, object]] = [
    {"k": 5, "metric": "euclidean", "p": 2, "weights": "uniform", "prob_thres": 0.45},
    {"k": 5, "metric": "euclidean", "p": 2, "weights": "inverted", "prob_thres": 0.45},
    {"k": 3, "metric": "euclidean", "p": 2, "weights": "uniform", "prob_thres": 0.45},
    {"k": 3, "metric": "euclidean", "p": 2, "weights": "inverted", "prob_thres": 0.45},
    {"k": 5, "metric": "manhattan", "p": 1, "weights": "inverted", "prob_thres": 0.45},
]


def get_knn_preprocessing_config(
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
) -> PreprocessingConfig:
    """Return the preprocessing configuration tailored for distance-based KNN."""
    return PreprocessingConfig(
        use_scaling=True,
        use_smote=True,
        use_feature_selection=True,
        use_pca=True,
        random_state=random_state,
        verbose=verbose,
    )


class KNN(BaseModel):
    """Batch-vectorized K-Nearest Neighbors classifier (from scratch).

    Reference: https://www.kaggle.com/code/fareselmenshawii/knn-from-scratch
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "minkowski",
        p: int = 2,
        batch_size: int = 256,
        weights: str = "uniform",
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")
        if metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. Expected one of {sorted(SUPPORTED_METRICS)}."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")
        if weights not in SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported weights '{weights}'. Expected one of {sorted(SUPPORTED_WEIGHTS)}."
            )

        self.k = k
        self.metric = metric
        self.batch_size = batch_size
        self.weights = weights
        if self.metric == "manhattan":
            self.p = 1
        elif self.metric == "euclidean":
            self.p = 2
        else:
            self.p = p

    def minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Minkowski distance (covers Manhattan p=1 and Euclidean p=2)."""
        return float(np.power(np.sum(np.abs(np.array(x1) - np.array(x2)) ** self.p), 1 / self.p))

    def hamming_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate normalized Hamming distance between two feature vectors."""
        return float(np.mean(x1 != x2))

    def _distances(self, Xb: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between batch Xb and self.X_train_."""
        n_feat = self.X_train_.shape[1]
        D = np.zeros((Xb.shape[0], self.X_train_.shape[0]))

        if self.metric == "hamming":
            for j in range(n_feat):
                D += Xb[:, j][:, None] != self.X_train_[:, j][None, :]
            return D / n_feat

        for j in range(n_feat):
            D += np.abs(Xb[:, j][:, None] - self.X_train_[:, j][None, :]) ** self.p
        return D ** (1 / self.p)

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNN:
        """Store training feature matrix, target labels, and unique classes."""
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X in memory-bounded batches."""
        if not hasattr(self, "X_train_") or not hasattr(self, "y_train_"):
            raise RuntimeError("KNN model must be fitted before calling predict().")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.shape[0] == 0:
            return np.empty(0, dtype=self.y_train_.dtype)

        preds = []
        for start in range(0, X_arr.shape[0], self.batch_size):
            Xb = X_arr[start : start + self.batch_size]
            D = self._distances(Xb)

            # argpartition finds k nearest indices faster than full sort
            idx = np.argpartition(D, self.k - 1, axis=1)[:, : self.k]
            labels = self.y_train_[idx]

            if self.weights == "inverted":
                k_dist = np.take_along_axis(D, idx, axis=1)
                w = 1.0 / (k_dist + 1e-5)
                scores = np.stack([(w * (labels == c)).sum(axis=1) for c in self.classes_], axis=1)
                preds.append(self.classes_[np.argmax(scores, axis=1)])
            else:
                counts = np.stack([(labels == c).sum(axis=1) for c in self.classes_], axis=1)
                preds.append(self.classes_[np.argmax(counts, axis=1)])

        return np.concatenate(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X in memory-bounded batches."""
        if not hasattr(self, "X_train_") or not hasattr(self, "y_train_"):
            raise RuntimeError("KNN model must be fitted before calling predict_proba().")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.shape[0] == 0:
            return np.empty((0, len(self.classes_)), dtype=float)

        probs = []
        for start in range(0, X_arr.shape[0], self.batch_size):
            Xb = X_arr[start : start + self.batch_size]
            D = self._distances(Xb)

            idx = np.argpartition(D, self.k - 1, axis=1)[:, : self.k]
            labels = self.y_train_[idx]

            if self.weights == "inverted":
                k_dist = np.take_along_axis(D, idx, axis=1)
                w = 1.0 / (k_dist + 1e-5)
                total_w = w.sum(axis=1, keepdims=True)
                scores = np.stack([(w * (labels == c)).sum(axis=1) for c in self.classes_], axis=1)
                probs.append(scores / total_w)
            else:
                counts = np.stack([(labels == c).sum(axis=1) for c in self.classes_], axis=1)
                probs.append(counts / self.k)

        return np.vstack(probs)
    def _predict(self, x: np.ndarray) -> object:
        """Predict the class label for a single sample (reference helper)."""
        if self.metric == "hamming":
            distances = np.array([self.hamming_distance(x, x_train) for x_train in self.X_train_])
        else:
            distances = np.array([self.minkowski_distance(x, x_train) for x_train in self.X_train_])

        k_idx_sorted = np.argsort(distances)[: self.k]
        labels = list(self.y_train_[k_idx_sorted])
        return max(labels, key=labels.count)


if __name__ == "__main__":
    import pandas as pd
    from src.preprocess.preprocess import run_preprocessing

    data_dir = Path(__file__).resolve().parents[2] / "data"
    raw_train = pd.read_csv(data_dir / "train.csv")
    raw_test = pd.read_csv(data_dir / "test.csv")

    prep = run_preprocessing(raw_train, raw_test, config=get_knn_preprocessing_config(verbose=False))
    knn = KNN(k=5, metric="euclidean").fit(prep.X_train[:500], prep.y_train[:500])
    sample_preds = knn.predict(prep.X_test[:10])
    print(f"KNN standalone smoke test OK -> sample predictions: {sample_preds.tolist()}")
