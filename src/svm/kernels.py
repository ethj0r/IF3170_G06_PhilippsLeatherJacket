"""Kernel functions for the SVM."""

import numpy as np


def _check(X, Y=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf")
    if Y is None:
        return X, X
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != X.shape[1]:
        raise ValueError("X and Y have incompatible shapes")
    if not np.isfinite(Y).all():
        raise ValueError("Y contains NaN or inf")
    return X, Y


def linear_kernel(X, Y=None):
    X, Y = _check(X, Y)
    return X @ Y.T


def polynomial_kernel(X, Y=None, degree=3, gamma=1.0, coef0=0.0):
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be >= 1")
    if gamma < 0:
        raise ValueError("gamma must be >= 0")

    X, Y = _check(X, Y)
    K = (gamma * (X @ Y.T) + coef0) ** degree
    if not np.isfinite(K).all():
        raise FloatingPointError("polynomial kernel overflowed")
    return K


def rbf_kernel(X, Y=None, gamma=1.0):
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    X, Y = _check(X, Y)
    n, m = X.shape[0], Y.shape[0]
    X2 = np.einsum("ij,ij->i", X, X)
    Y2 = np.einsum("ij,ij->i", Y, Y)

    # For large inputs, fill the kernel in place so we do not allocate
    # several full n x m temporaries.
    if n * m * 8 > 256 * 1024 * 1024:
        K = X @ Y.T
        K *= -2.0
        rows = max(1, (64 * 1024 * 1024) // (8 * m))
        for start in range(0, n, rows):
            end = min(start + rows, n)
            block = K[start:end]
            block += X2[start:end, None]
            block += Y2[None, :]
            np.maximum(block, 0.0, out=block)
            block *= -gamma
            np.exp(block, out=block)
        return K

    D = X2[:, None] + Y2[None, :] - 2.0 * (X @ Y.T)
    np.maximum(D, 0.0, out=D)
    return np.exp(-gamma * D)


def kernel_matrix(X, Y=None, kernel="rbf", degree=3, gamma=1.0, coef0=0.0):
    kernel = kernel.lower()
    if kernel == "linear":
        return linear_kernel(X, Y)
    if kernel in ("poly", "polynomial"):
        return polynomial_kernel(X, Y, degree, gamma, coef0)
    if kernel in ("rbf", "gaussian"):
        return rbf_kernel(X, Y, gamma)
    raise ValueError(f"unknown kernel: {kernel}")


def resolve_gamma(gamma, X, kernel):
    if isinstance(gamma, str):
        X = np.asarray(X, dtype=float)
        if gamma == "scale":
            var = X.var()
            if var <= 0:
                var = np.finfo(float).eps
            return 1.0 / (X.shape[1] * var)
        if gamma == "auto":
            return 1.0 / X.shape[1]
        raise ValueError(f"unknown gamma: {gamma}")

    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    return gamma
