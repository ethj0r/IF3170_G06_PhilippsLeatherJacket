from pathlib import Path

import joblib
import numpy as np


def is_missing(value):
    return value is None or value != value


class NaiveBayes:
    def __init__(self, categorical_features=None, alpha=1.0, var_smoothing=1e-9):
        self.categorical_features = categorical_features
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def _split_columns(self, n_features):
        cat = [] if self.categorical_features is None else list(self.categorical_features)
        num = [j for j in range(n_features) if j not in cat]
        return cat, num

    def _encode(self, column, j):
        mapping = self.value_index_[j]
        unseen = len(mapping)
        codes = [-1 if is_missing(v) else mapping.get(v, unseen) for v in column]
        return np.array(codes, dtype=int)

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        self.cat_, self.num_ = self._split_columns(self.n_features_)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
        self.class_log_prior_ = np.log(counts / counts.sum())

        X_num = X[:, self.num_].astype(float)
        self.theta_ = np.zeros((n_classes, len(self.num_)))
        self.var_ = np.zeros((n_classes, len(self.num_)))
        for k, c in enumerate(self.classes_):
            rows = X_num[y == c]
            self.theta_[k] = np.nanmean(rows, axis=0)
            self.var_[k] = np.nanvar(rows, axis=0)
        if len(self.num_) > 0:
            self.epsilon_ = self.var_smoothing * np.nanvar(X_num, axis=0).max()
        else:
            self.epsilon_ = 0.0
        self.var_ += self.epsilon_

        self.value_index_ = {}
        self.feature_log_prob_ = {}
        for j in self.cat_:
            values = sorted({v for v in X[:, j] if not is_missing(v)}, key=str)
            self.value_index_[j] = {v: i for i, v in enumerate(values)}
            codes = self._encode(X[:, j], j)
            n_values = len(values) + 1
            table = np.zeros((n_classes, n_values))
            for k, c in enumerate(self.classes_):
                known = codes[(y == c) & (codes >= 0)]
                freq = np.bincount(known, minlength=n_values).astype(float)
                table[k] = np.log((freq + self.alpha) / (len(known) + self.alpha * n_values))
            self.feature_log_prob_[j] = table
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=object)
        jll = np.tile(self.class_log_prior_, (X.shape[0], 1))

        X_num = X[:, self.num_].astype(float)
        for k in range(len(self.classes_)):
            log_pdf = -0.5 * np.log(2 * np.pi * self.var_[k]) - (X_num - self.theta_[k]) ** 2 / (2 * self.var_[k])
            jll[:, k] += np.nansum(log_pdf, axis=1)

        for j in self.cat_:
            codes = self._encode(X[:, j], j)
            known = codes >= 0
            jll[known] += self.feature_log_prob_[j][:, codes[known]].T
        return jll

    def feature_log_ratio(self, X):
        X = np.asarray(X, dtype=object)
        ratio = np.zeros((X.shape[0], self.n_features_))

        X_num = X[:, self.num_].astype(float)
        log_pdf = [
            -0.5 * np.log(2 * np.pi * self.var_[k]) - (X_num - self.theta_[k]) ** 2 / (2 * self.var_[k])
            for k in range(2)
        ]
        ratio[:, self.num_] = np.nan_to_num(log_pdf[1] - log_pdf[0])

        for j in self.cat_:
            codes = self._encode(X[:, j], j)
            known = codes >= 0
            table = self.feature_log_prob_[j]
            ratio[known, j] = table[1, codes[known]] - table[0, codes[known]]
        return ratio

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        top = jll.max(axis=1, keepdims=True)
        log_norm = top + np.log(np.exp(jll - top).sum(axis=1, keepdims=True))
        return jll - log_norm

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(Path(path))


class WeightedNaiveBayes(NaiveBayes):
    def __init__(self, categorical_features=None, alpha=1.0, var_smoothing=1e-9,
                 l2=1.0, n_folds=5, max_iter=50, tol=1e-8, random_state=None):
        super().__init__(categorical_features, alpha, var_smoothing)
        self.l2 = l2
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        fold = rng.permutation(len(y)) % self.n_folds

        ratio = np.zeros((len(y), X.shape[1]))
        for f in range(self.n_folds):
            inner = NaiveBayes(self.categorical_features, self.alpha, self.var_smoothing)
            inner.fit(X[fold != f], y[fold != f])
            ratio[fold == f] = inner.feature_log_ratio(X[fold == f])

        super().fit(X, y)
        target = (y == self.classes_[1]).astype(float)
        self.bias_, self.weights_ = self._fit_weights(ratio, target)
        return self

    def _fit_weights(self, ratio, target):
        A = np.hstack([np.ones((len(ratio), 1)), ratio])
        w = np.zeros(A.shape[1])
        penalty = self.l2 * np.eye(A.shape[1])
        penalty[0, 0] = 0.0
        for _ in range(self.max_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -500, 500)))
            grad = A.T @ (p - target) + penalty @ w
            hessian = (A * (p * (1 - p))[:, None]).T @ A + penalty
            step = np.linalg.solve(hessian, grad)
            w -= step
            if np.abs(step).max() < self.tol:
                break
        return w[0], w[1:]

    def decision_function(self, X):
        return self.bias_ + self.feature_log_ratio(X) @ self.weights_

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-np.clip(self.decision_function(X), -500, 500)))
        return np.column_stack([1 - p, p])

    def predict_log_proba(self, X):
        return np.log(np.clip(self.predict_proba(X), 1e-300, None))

    def predict(self, X):
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]
