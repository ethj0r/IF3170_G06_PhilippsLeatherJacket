"""Binary SVM trained with SMO."""

import numpy as np

from .kernels import kernel_matrix, resolve_gamma
from .smo import KernelProvider, solve_smo


class SVM:
    """Soft-margin binary SVM with linear, polynomial, or RBF kernels."""

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        max_iter=50000,
        random_state=None,
        cache_size_mb=512.0,
        verbose=False,
    ):
        self.C = C
        self.kernel = kernel.lower()
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.cache_size_mb = cache_size_mb
        self.verbose = verbose

    def _check_params(self):
        if self.C <= 0:
            raise ValueError("C must be > 0")
        if self.kernel not in ("linear", "poly", "rbf"):
            raise ValueError(f"unknown kernel: {self.kernel}")
        if int(self.degree) != self.degree or self.degree < 1:
            raise ValueError("degree must be a positive integer")
        if self.tol <= 0:
            raise ValueError("tol must be > 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.cache_size_mb <= 0:
            raise ValueError("cache_size_mb must be > 0")

    def fit(self, X, y):
        self._check_params()
        for name in ("coef_", "margin_"):
            if hasattr(self, name):
                delattr(self, name)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if len(X) != len(y):
            raise ValueError("X and y have different lengths")
        if len(X) == 0:
            raise ValueError("empty training set")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("SVM only supports binary classification")

        # SMO works with labels -1 and +1.
        y_pm = np.where(y == classes[1], 1.0, -1.0)
        gamma = resolve_gamma(self.gamma, X, self.kernel)

        provider = KernelProvider(
            X,
            self.kernel,
            int(self.degree),
            gamma,
            float(self.coef0),
            float(self.cache_size_mb),
        )
        result = solve_smo(
            provider,
            y_pm,
            float(self.C),
            tol=float(self.tol),
            max_iter=int(self.max_iter),
            random_state=self.random_state,
            verbose=self.verbose,
        )

        sv = result.alpha > 1e-5
        if not sv.any():
            sv = np.ones(len(X), dtype=bool)
        sv_idx = np.flatnonzero(sv)

        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.gamma_ = float(gamma)
        self.alpha_ = result.alpha
        self.support_ = sv_idx
        self.support_indices_ = sv_idx
        self.support_vectors_ = X[sv]
        self.dual_coef_ = (result.alpha[sv] * y_pm[sv]).reshape(1, -1)
        self.intercept_ = np.array([result.b])
        self.n_support_ = np.array(
            [
                int(np.sum(sv & (y_pm == -1))),
                int(np.sum(sv & (y_pm == 1))),
            ]
        )
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        self.kkt_violation_ = result.kkt_violation
        self.fit_status_ = 0 if result.converged else 1

        if self.kernel == "linear":
            self.coef_ = self.dual_coef_ @ self.support_vectors_
            norm = np.linalg.norm(self.coef_)
            self.margin_ = 2.0 / norm if norm > 0 else np.inf

        return self

    def _check_fitted(self):
        if not hasattr(self, "support_vectors_"):
            raise RuntimeError("fit the model first")

    def decision_function(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong shape")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf")

        K = kernel_matrix(
            X,
            self.support_vectors_,
            self.kernel,
            int(self.degree),
            self.gamma_,
            float(self.coef0),
        )
        return K @ self.dual_coef_[0] + self.intercept_[0]

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores > 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def __repr__(self):
        if hasattr(self, "support_vectors_"):
            return (
                f"SVM(C={self.C}, kernel={self.kernel!r}, "
                f"n_support={len(self.support_vectors_)}, "
                f"converged={self.converged_})"
            )
        return f"SVM(C={self.C}, kernel={self.kernel!r}, gamma={self.gamma!r})"
