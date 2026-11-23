"""Tests for the from-scratch SVM package."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import (
    linear_kernel as sk_linear_kernel,
    polynomial_kernel as sk_polynomial_kernel,
    rbf_kernel as sk_rbf_kernel,
)
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.svm.kernels import (  # noqa: E402
    kernel_matrix,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
)
from src.svm.model import SVM  # noqa: E402


class TestKernels(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(0)
        self.X = rng.normal(size=(12, 4))
        self.Y = rng.normal(size=(7, 4))

    def test_linear_matches_sklearn(self) -> None:
        np.testing.assert_allclose(
            linear_kernel(self.X, self.Y),
            sk_linear_kernel(self.X, self.Y),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_polynomial_matches_sklearn(self) -> None:
        np.testing.assert_allclose(
            polynomial_kernel(
                self.X, self.Y, degree=3, gamma=0.5, coef0=1.5
            ),
            sk_polynomial_kernel(
                self.X, self.Y, degree=3, gamma=0.5, coef0=1.5
            ),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rbf_matches_sklearn(self) -> None:
        np.testing.assert_allclose(
            rbf_kernel(self.X, self.Y, gamma=0.7),
            sk_rbf_kernel(self.X, self.Y, gamma=0.7),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_kernel_matrix_dispatch(self) -> None:
        np.testing.assert_allclose(
            kernel_matrix(self.X, None, kernel="linear"),
            sk_linear_kernel(self.X),
        )
        np.testing.assert_allclose(
            kernel_matrix(self.X, None, kernel="rbf", gamma=0.3),
            sk_rbf_kernel(self.X, gamma=0.3),
        )


class TestSVM(unittest.TestCase):
    @staticmethod
    def _separable_data() -> tuple[np.ndarray, np.ndarray]:
        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [3.0, 3.0],
                [3.0, 4.0],
                [4.0, 3.0],
                [4.0, 4.0],
            ]
        )
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        return X, y

    def test_linearly_separable(self) -> None:
        X, y = self._separable_data()
        model = SVM(
            C=1e6,
            kernel="linear",
            tol=1e-4,
            max_iter=300,
            random_state=0,
        ).fit(X, y)

        self.assertGreaterEqual(model.score(X, y), 0.999)
        self.assertTrue(np.all(model.alpha_ >= 0.0))
        self.assertTrue(np.all(model.alpha_ <= 1e6))
        y_pm = np.where(y == model.classes_[1], 1.0, -1.0)
        self.assertAlmostEqual(float(np.sum(model.alpha_ * y_pm)), 0.0, places=6)
        self.assertLessEqual(model.kkt_violation_, 1e-2)
        self.assertGreater(model.margin_, 0.0)
        self.assertGreater(model.support_vectors_.shape[0], 0)

    def test_soft_margin_alpha_bounds(self) -> None:
        rng = np.random.RandomState(7)
        X = rng.normal(size=(50, 2))
        y = (X[:, 0] + X[:, 1] + 0.4 * rng.normal(size=50) > 0).astype(int)

        for C in (0.05, 10.0):
            model = SVM(
                C=C,
                kernel="linear",
                tol=1e-3,
                max_iter=100,
                random_state=1,
            ).fit(X, y)
            self.assertTrue(np.all(model.alpha_ >= -1e-12))
            self.assertTrue(np.all(model.alpha_ <= C + 1e-12))
            y_pm = np.where(y == model.classes_[1], 1.0, -1.0)
            self.assertAlmostEqual(float(np.sum(model.alpha_ * y_pm)), 0.0, places=5)
            self.assertTrue(np.all(np.isfinite(model.decision_function(X))))
            self.assertTrue(model.converged_)
            self.assertLessEqual(model.kkt_violation_, 1e-2)

    def test_xor_kernels(self) -> None:
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1, 1, 0])

        linear = SVM(C=10.0, kernel="linear", max_iter=100, random_state=0).fit(
            X, y
        )
        rbf = SVM(
            C=10.0,
            kernel="rbf",
            gamma=2.0,
            max_iter=100,
            random_state=0,
        ).fit(X, y)
        poly = SVM(
            C=10.0,
            kernel="poly",
            degree=2,
            gamma=1.0,
            coef0=1.0,
            max_iter=100,
            random_state=0,
        ).fit(X, y)

        self.assertLessEqual(linear.score(X, y), 0.75)
        self.assertGreaterEqual(rbf.score(X, y), 0.999)
        self.assertGreaterEqual(poly.score(X, y), 0.999)

    def test_support_vectors_lie_on_margin(self) -> None:
        X, y = self._separable_data()
        model = SVM(
            C=1e6,
            kernel="linear",
            tol=1e-4,
            max_iter=300,
            random_state=0,
        ).fit(X, y)
        scores = model.decision_function(model.support_vectors_)
        np.testing.assert_allclose(np.abs(scores), 1.0, atol=2e-2)

    def test_unseen_linear_prediction(self) -> None:
        X, y = self._separable_data()
        model = SVM(C=10.0, kernel="linear", max_iter=100, random_state=0).fit(X, y)
        X_new = np.array([[2.0, 2.0], [0.5, 0.5], [3.5, 3.5]])
        manual = X_new @ model.coef_[0] + model.intercept_[0]
        np.testing.assert_allclose(model.decision_function(X_new), manual, atol=1e-10)

    def test_serialization_roundtrip(self) -> None:
        X, y = self._separable_data()
        model = SVM(C=1.0, kernel="rbf", gamma="scale", random_state=0).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.pkl"
            joblib.dump(model, path)
            loaded = joblib.load(path)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
        self.assertEqual(loaded.kernel, model.kernel)
        self.assertAlmostEqual(loaded.gamma_, model.gamma_, places=12)

    def test_reproducibility(self) -> None:
        X, y = self._separable_data()
        kwargs = dict(C=1.0, kernel="rbf", gamma=0.5, random_state=123)
        first = SVM(**kwargs).fit(X, y)
        second = SVM(**kwargs).fit(X, y)
        np.testing.assert_array_equal(first.alpha_, second.alpha_)
        np.testing.assert_array_equal(first.predict(X), second.predict(X))

    def test_gamma_scale_matches_sklearn(self) -> None:
        X, y = self._separable_data()
        model = SVM(C=1.0, kernel="rbf", gamma="scale", random_state=0).fit(X, y)
        expected = 1.0 / (X.shape[1] * X.var())
        self.assertAlmostEqual(model.gamma_, expected, places=12)

        svc = SVC(C=1.0, kernel="rbf", gamma="scale").fit(X, y)
        self.assertAlmostEqual(model.gamma_, float(svc._gamma), places=12)

    def test_against_sklearn(self) -> None:
        X, y = make_blobs(
            n_samples=80,
            centers=2,
            cluster_std=1.5,
            random_state=42,
        )
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        scratch = SVM(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            tol=1e-3,
            max_iter=100,
            random_state=0,
        ).fit(X, y)
        svc = SVC(C=1.0, kernel="rbf", gamma="scale").fit(X, y)

        scratch_score = scratch.score(X, y)
        svc_score = svc.score(X, y)
        self.assertLess(abs(scratch_score - svc_score), 0.15)

        agreement = np.mean(
            np.sign(scratch.decision_function(X)) == np.sign(svc.decision_function(X))
        )
        self.assertGreater(agreement, 0.85)

    def test_duplicate_points_are_finite(self) -> None:
        X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        y = np.array([0, 0, 1, 1])
        model = SVM(C=1.0, kernel="linear", max_iter=50, random_state=0).fit(X, y)
        scores = model.decision_function(X)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(np.isfinite(model.alpha_)))

    def test_multiclass_rejected(self) -> None:
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            SVM(C=1.0, kernel="linear").fit(X, y)


if __name__ == "__main__":
    unittest.main()
