"""Binary Logistic Regression using Maximum Likelihood Estimation, SGA, and Adam."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib

    _HAS_JOBLIB = True
except ImportError:
    joblib = None
    _HAS_JOBLIB = False

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.models.base import BaseModel
    from src.preprocess.config import DEFAULT_RANDOM_STATE, PreprocessingConfig
else:
    from .base import BaseModel
    from ..preprocess.config import DEFAULT_RANDOM_STATE, PreprocessingConfig

DEFAULT_LOGREG_CONFIGS: list[dict[str, Any]] = [
    {"optimizer": "sga", "penalty": None, "learning_rate": 0.01, "reg_lambda": 0.0, "l1_ratio": 0.5, "max_iter": 20, "random_state": DEFAULT_RANDOM_STATE},
    {"optimizer": "sga", "penalty": "l1", "learning_rate": 0.01, "reg_lambda": 1e-4, "l1_ratio": 0.5, "max_iter": 20, "random_state": DEFAULT_RANDOM_STATE},
    {"optimizer": "sga", "penalty": "l2", "learning_rate": 0.01, "reg_lambda": 1e-4, "l1_ratio": 0.5, "max_iter": 20, "random_state": DEFAULT_RANDOM_STATE},
    {"optimizer": "sga", "penalty": "elasticnet", "learning_rate": 0.01, "reg_lambda": 1e-4, "l1_ratio": 0.5, "max_iter": 20, "random_state": DEFAULT_RANDOM_STATE},
    {"optimizer": "adam", "penalty": "l2", "learning_rate": 0.02, "batch_size": 256, "reg_lambda": 1e-4, "l1_ratio": 0.5, "max_iter": 30, "random_state": DEFAULT_RANDOM_STATE},
    {"optimizer": "adam", "penalty": "elasticnet", "learning_rate": 0.02, "batch_size": 256, "reg_lambda": 1e-4, "l1_ratio": 0.5, "max_iter": 30, "random_state": DEFAULT_RANDOM_STATE},
]


def get_logreg_preprocessing_config(
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = True,
) -> PreprocessingConfig:
    """Return the preprocessing configuration tailored for Logistic Regression."""
    return PreprocessingConfig(
        use_scaling=True,
        use_smote=True,
        use_feature_selection=True,
        use_pca=True,
        random_state=random_state,
        verbose=verbose,
    )


class LogisticRegression(BaseModel):
    """Binary Logistic Regression classifier trained with SGA or mini-batch Adam.

    References:
    - M. L. Khodra (2021), Logistic Regression: Stochastic Gradient Ascent.
    - D. P. Kingma and J. Ba (2015), Adam: A Method for Stochastic Optimization,
      https://arxiv.org/abs/1412.6980
    """

    VALID_OPTIMIZERS = ("sga", "adam")

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        penalty: str | None = None,
        reg_lambda: float = 0.01,
        l1_ratio: float = 0.5,
        threshold: float = 0.5,
        shuffle: bool = True,
        tol: float | None = 1e-6,
        random_state: int | None = None,
        verbose: bool = False,
        optimizer: str = "sga",
        batch_size: int | None = 256,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Initialize model hyperparameters and training state."""
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.penalty = penalty.lower() if isinstance(penalty, str) else penalty
        self.reg_lambda = reg_lambda
        self.l1_ratio = l1_ratio
        self.threshold = threshold
        self.shuffle = shuffle
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.optimizer = optimizer.lower() if isinstance(optimizer, str) else "sga"
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.b_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.lcl_history_: list[float] = []
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    def _validate_hyperparameters(self) -> None:
        """Validate initialization hyperparameters before training."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.penalty not in (None, "none", "l1", "l2", "elasticnet"):
            raise ValueError("penalty must be None, 'none', 'l1', 'l2', or 'elasticnet'")
        if self.reg_lambda < 0:
            raise ValueError("reg_lambda must be >= 0")
        if not (0.0 <= self.l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must be in (0, 1)")
        if self.optimizer not in self.VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {self.VALID_OPTIMIZERS}")

    @staticmethod
    def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
        """Compute numerically stable sigmoid activation 1 / (1 + exp(-z))."""
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        """Prepend bias column x_0 = 1 to feature matrix X."""
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack((ones, X))

    @property
    def intercept_(self) -> float | None:
        """Return fitted bias weight b_0."""
        if self.b_ is None:
            return None
        return float(self.b_[0])

    @property
    def coef_(self) -> np.ndarray | None:
        """Return fitted feature weights (b_1, ..., b_d)."""
        if self.b_ is None:
            return None
        return self.b_[1:].copy()

    def log_odds(self, X: np.ndarray) -> np.ndarray:
        """Compute linear combination (log-odds) b^T x."""
        if self.b_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_bias = self._add_intercept(X_arr)
        return X_bias @ self.b_

    def _compute_lcl(self, X_bias: np.ndarray, y: np.ndarray) -> float:
        """Compute Log Joint Conditional Likelihood (LCL) on the dataset."""
        eps = 1e-15
        p = self._sigmoid(X_bias @ self.b_)
        p_clipped = np.clip(p, eps, 1.0 - eps)
        lcl = np.sum(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
        return float(lcl)

    def _compute_penalty(self) -> float:
        """Compute regularization penalty term on feature weights (excluding bias)."""
        if self.penalty in (None, "none") or self.reg_lambda == 0.0:
            return 0.0
        w = self.b_[1:]
        if self.penalty == "l1":
            return float(self.reg_lambda * np.sum(np.abs(w)))
        if self.penalty == "l2":
            return float(0.5 * self.reg_lambda * np.sum(w ** 2))
        l1_term = self.l1_ratio * np.sum(np.abs(w))
        l2_term = 0.5 * (1.0 - self.l1_ratio) * np.sum(w ** 2)
        return float(self.reg_lambda * (l1_term + l2_term))

    def _regularization_grad(self) -> np.ndarray:
        """Compute gradient of the regularization penalty with respect to feature weights."""
        w = self.b_[1:]
        if self.penalty == "l1":
            return self.reg_lambda * np.sign(w)
        if self.penalty == "l2":
            return self.reg_lambda * w
        if self.penalty == "elasticnet":
            l1_grad = self.l1_ratio * np.sign(w)
            l2_grad = (1.0 - self.l1_ratio) * w
            return self.reg_lambda * (l1_grad + l2_grad)
        return np.zeros_like(w)

    def _record_epoch(self, epoch: int, X_bias: np.ndarray, y_bin: np.ndarray) -> bool:
        """Record epoch LCL and log-loss; return True when convergence tolerance is met."""
        current_lcl = self._compute_lcl(X_bias, y_bin)
        current_loss = (-current_lcl / X_bias.shape[0]) + self._compute_penalty()
        self.lcl_history_.append(current_lcl)
        self.loss_history_.append(current_loss)
        self.n_iter_ = epoch

        if self.verbose and (epoch == 1 or epoch % 10 == 0 or epoch == self.max_iter):
            print(f"[Epoch {epoch:4d}/{self.max_iter}] LCL: {current_lcl:.4f} | Loss: {current_loss:.6f}")

        if self.tol is not None and len(self.loss_history_) >= 2:
            if abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {epoch}")
                return True
        return False

    def _fit_sga(self, X_bias: np.ndarray, y_bin: np.ndarray) -> None:
        """Train using Stochastic Gradient Ascent."""
        n_samples = X_bias.shape[0]
        rng = np.random.default_rng(self.random_state)
        use_reg = self.penalty not in (None, "none") and self.reg_lambda > 0.0

        for epoch in range(1, self.max_iter + 1):
            indices = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
            for idx in indices:
                x_i = X_bias[idx]
                p_i = self._sigmoid(np.dot(self.b_, x_i))
                self.b_ += self.learning_rate * (y_bin[idx] - p_i) * x_i
                if use_reg:
                    self.b_[1:] -= self.learning_rate * self._regularization_grad()

            if self._record_epoch(epoch, X_bias, y_bin):
                break

    def _fit_adam(self, X_bias: np.ndarray, y_bin: np.ndarray) -> None:
        """Train using mini-batch Adam optimizer."""
        n_samples, n_params = X_bias.shape
        rng = np.random.default_rng(self.random_state)
        use_reg = self.penalty not in (None, "none") and self.reg_lambda > 0.0
        bs = n_samples if (self.batch_size is None or self.batch_size >= n_samples) else max(1, int(self.batch_size))

        m = np.zeros(n_params, dtype=np.float64)
        v = np.zeros(n_params, dtype=np.float64)
        t_step = 0

        for epoch in range(1, self.max_iter + 1):
            indices = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
            for start in range(0, n_samples, bs):
                batch_idx = indices[start : start + bs]
                Xb, yb = X_bias[batch_idx], y_bin[batch_idx]

                p_b = self._sigmoid(Xb @ self.b_)
                grad = (Xb.T @ (p_b - yb)) / float(len(batch_idx))
                if use_reg:
                    grad[1:] += self._regularization_grad()

                t_step += 1
                m = self.beta1 * m + (1.0 - self.beta1) * grad
                v = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
                m_hat = m / (1.0 - self.beta1 ** t_step)
                v_hat = v / (1.0 - self.beta2 ** t_step)
                self.b_ -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            if self._record_epoch(epoch, X_bias, y_bin):
                break

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """Fit binary Logistic Regression model using SGA or mini-batch Adam."""
        self._validate_hyperparameters()

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).ravel()
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.classes_ = np.unique(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(f"Expected 2 classes, got {len(self.classes_)}")

        y_bin = (y_arr == self.classes_[1]).astype(np.float64)
        X_bias = self._add_intercept(X_arr)

        self.b_ = np.zeros(X_bias.shape[1], dtype=np.float64)
        self.lcl_history_ = []
        self.loss_history_ = []
        self.n_iter_ = 0

        if self.optimizer == "adam":
            self._fit_adam(X_bias, y_bin)
        else:
            self._fit_sga(X_bias, y_bin)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for [class 0, class 1]."""
        z = self.log_odds(X)
        p_pos = np.asarray(self._sigmoid(z), dtype=np.float64)
        return np.column_stack((1.0 - p_pos, p_pos))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using the configured decision threshold."""
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted yet.")
        p_pos = self.predict_proba(X)[:, 1]
        pred_indices = (p_pos >= self.threshold).astype(int)
        return self.classes_[pred_indices]

    def save(self, filepath: str | Path) -> None:
        """Save the fitted model instance to disk using joblib or pickle fallback."""
        if self.b_ is None:
            raise RuntimeError("Model is not fitted yet.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_JOBLIB:
            joblib.dump(self, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> LogisticRegression:
        """Load a saved LogisticRegression model instance from disk."""
        path = Path(filepath)
        if _HAS_JOBLIB:
            model = joblib.load(path)
        else:
            with open(path, "rb") as f:
                model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__} instance, got {type(model).__name__}")
        return model


if __name__ == "__main__":
    rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
    X_demo = rng.normal(size=(200, 4))
    y_demo = (X_demo[:, 0] - 0.8 * X_demo[:, 1] + 0.2 * rng.normal(size=200) > 0).astype(int)
    model = LogisticRegression(optimizer="adam", penalty="elasticnet", reg_lambda=1e-4, max_iter=30, random_state=DEFAULT_RANDOM_STATE)
    model.fit(X_demo, y_demo)
    print(f"LogisticRegression standalone smoke test OK -> sample predictions: {model.predict(X_demo[:10]).tolist()}")
