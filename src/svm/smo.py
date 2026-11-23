"""SMO solver for the SVM dual.

    max sum(alpha) - 1/2 alpha^T Q alpha
    s.t. 0 <= alpha_i <= C, sum(alpha_i y_i) = 0

with Q_ij = y_i y_j K(x_i, x_j). Uses second-order working set selection.

References:
- Platt (1998), Sequential Minimal Optimization.
- Fan, Chen, Lin (2005), Working Set Selection Using Second Order Information.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .kernels import kernel_matrix


class KernelProvider:
    """Kernel rows from a full matrix or an LRU cache."""

    def __init__(self, X, kernel, degree, gamma, coef0, cache_size_mb=512.0):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.kernel = kernel.lower()
        self.degree = int(degree)
        self.gamma = float(gamma)
        self.coef0 = float(coef0)
        self.n = len(self.X)

        full_bytes = self.n * self.n * 8
        cache_bytes = max(1, int(cache_size_mb * 1024 * 1024))

        self.full = None
        self._cache = OrderedDict()
        if full_bytes <= cache_bytes:
            self.full = kernel_matrix(
                self.X, None, self.kernel, self.degree, self.gamma, self.coef0
            )
            self.max_rows = self.n
        else:
            self.max_rows = max(1, cache_bytes // (self.n * 8))

        self._diag = self._compute_diag()

    def _compute_diag(self):
        sq = np.einsum("ij,ij->i", self.X, self.X)
        if self.kernel == "linear":
            return sq.copy()
        if self.kernel in ("poly", "polynomial"):
            return (self.gamma * sq + self.coef0) ** self.degree
        if self.kernel in ("rbf", "gaussian"):
            return np.ones(self.n)
        raise ValueError(f"unknown kernel: {self.kernel}")

    def row(self, i):
        if self.full is not None:
            return self.full[i]

        row = self._cache.get(i)
        if row is not None:
            self._cache.move_to_end(i)
            return row

        row = kernel_matrix(
            self.X[i : i + 1], self.X, self.kernel, self.degree, self.gamma, self.coef0
        )[0]
        self._cache[i] = row
        if len(self._cache) > self.max_rows:
            self._cache.popitem(last=False)
        return row

    def diag(self, i):
        return float(self._diag[i])

    def diag_array(self):
        return self._diag


@dataclass
class SMOResult:
    alpha: np.ndarray
    b: float
    errors: np.ndarray
    converged: bool
    n_iter: int
    kkt_violation: float


def _decision_values(provider, alpha, y, b):
    ay = alpha * y
    if provider.full is not None:
        return provider.full @ ay + b
    return np.array([provider.row(i) @ ay + b for i in range(provider.n)])


def _kkt_violation(y, alpha, C_vec, f):
    # E_i = f_i - y_i; KKT is max(E[low]) <= min(E[up]).
    errors = f - y
    up = ((y == 1) & (alpha < C_vec)) | ((y == -1) & (alpha > 0))
    low = ((y == 1) & (alpha > 0)) | ((y == -1) & (alpha < C_vec))

    min_up = np.min(errors[up]) if up.any() else 0.0
    max_low = np.max(errors[low]) if low.any() else 0.0
    return max(0.0, float(max_low - min_up))


def solve_smo(
    provider,
    y,
    C,
    tol=1e-3,
    max_iter=50000,
    eps=1e-10,
    random_state=None,
    verbose=False,
):
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if provider.n != n:
        raise ValueError("provider and labels have different sizes")
    if not np.all((y == 1) | (y == -1)):
        raise ValueError("labels must be -1 or +1")

    if np.isscalar(C):
        C_vec = np.full(n, float(C))
    else:
        C_vec = np.asarray(C, dtype=float).ravel()
        if len(C_vec) != n:
            raise ValueError("C must have one value per sample")
    if np.any(C_vec <= 0):
        raise ValueError("C must be > 0")

    rng = np.random.RandomState(random_state)
    alpha = np.zeros(n)
    b = 0.0
    converged = False
    kkt_violation = np.inf
    n_iter = 0
    alpha_tol = max(1e-10, eps * 1e-2)
    diag = provider.diag_array()
    recompute_every = 100
    max_candidates = 16

    def clean_bounds():
        alpha[alpha < alpha_tol] = 0.0
        near = alpha > C_vec - alpha_tol
        alpha[near] = C_vec[near]

    clean_bounds()
    errors = _decision_values(provider, alpha, y, b) - y

    for iteration in range(1, max_iter + 1):
        clean_bounds()
        if iteration % recompute_every == 0:
            errors = _decision_values(provider, alpha, y, b) - y

        up = ((y == 1) & (alpha < C_vec - alpha_tol)) | (
            (y == -1) & (alpha > alpha_tol)
        )
        low = ((y == 1) & (alpha > alpha_tol)) | (
            (y == -1) & (alpha < C_vec - alpha_tol)
        )
        if not up.any() or not low.any():
            converged = True
            break

        up_idx = np.flatnonzero(up)
        low_idx = np.flatnonzero(low)
        i = int(up_idx[np.argmin(errors[up_idx])])
        kkt_violation = max(0.0, float(np.max(errors[low_idx])) - float(errors[i]))
        if kkt_violation <= tol:
            converged = True
            break

        Ei = float(errors[i])
        yi = float(y[i])
        ai = float(alpha[i])
        Kii = float(diag[i])
        row_i = provider.row(i)

        # Second-order pair selection for i.
        diffs = errors[low_idx] - Ei
        positive = diffs > 0
        scores = np.full(len(low_idx), -np.inf)
        if positive.any():
            cand = low_idx[positive]
            d = diffs[positive]
            quad = Kii + diag[cand] - 2.0 * row_i[cand]
            valid = quad > 1e-12
            if valid.any():
                sc = np.full(len(cand), -np.inf)
                sc[valid] = d[valid] ** 2 / quad[valid]
                scores[positive] = sc
                order = np.argsort(scores)[::-1]
                order = order[np.isfinite(scores[order])][:max_candidates]
            else:
                order = np.array([], dtype=int)
        else:
            order = np.array([], dtype=int)

        updated = False
        for pos in order:
            j = int(low_idx[pos])
            if j == i:
                continue
            Ej = float(errors[j])
            yj = float(y[j])
            aj = float(alpha[j])
            Kjj = float(diag[j])
            Kij = float(row_i[j])
            eta = 2.0 * Kij - Kii - Kjj
            if eta >= -1e-12:
                continue

            if yi != yj:
                L = max(0.0, aj - ai)
                H = min(C_vec[j], C_vec[i] + aj - ai)
            else:
                L = max(0.0, ai + aj - C_vec[i])
                H = min(C_vec[j], ai + aj)
            if L >= H:
                continue

            aj_new = aj - yj * (Ei - Ej) / eta
            aj_new = min(max(aj_new, L), H)
            if aj_new < alpha_tol:
                aj_new = 0.0
            elif aj_new > C_vec[j] - alpha_tol:
                aj_new = C_vec[j]
            if abs(aj_new - aj) < eps:
                continue

            ai_new = ai + yi * yj * (aj - aj_new)
            ai_new = min(max(ai_new, 0.0), C_vec[i])
            if ai_new < alpha_tol:
                ai_new = 0.0
            elif ai_new > C_vec[i] - alpha_tol:
                ai_new = C_vec[i]

            dai = ai_new - ai
            daj = aj_new - aj
            if abs(dai) < eps and abs(daj) < eps:
                continue

            b1 = b - Ei - yi * dai * Kii - yj * daj * Kij
            b2 = b - Ej - yi * dai * Kij - yj * daj * Kjj
            if 0 < ai_new < C_vec[i]:
                b_new = b1
            elif 0 < aj_new < C_vec[j]:
                b_new = b2
            else:
                b_new = 0.5 * (b1 + b2)

            errors += yi * dai * row_i
            errors += yj * daj * provider.row(j)
            errors += b_new - b

            alpha[i] = ai_new
            alpha[j] = aj_new
            b = float(b_new)
            updated = True
            break

        if not updated:
            # Try random pairs if the best candidates cannot be updated.
            for _ in range(200):
                i_r = int(rng.choice(up_idx))
                j_r = int(rng.choice(low_idx))
                if i_r == j_r:
                    continue
                Ei_r = float(errors[i_r])
                Ej_r = float(errors[j_r])
                if Ej_r <= Ei_r:
                    continue
                yi_r = float(y[i_r])
                yj_r = float(y[j_r])
                ai_r = float(alpha[i_r])
                aj_r = float(alpha[j_r])
                Kii_r = float(diag[i_r])
                Kjj_r = float(diag[j_r])
                row_r = provider.row(i_r)
                Kij_r = float(row_r[j_r])
                eta_r = 2.0 * Kij_r - Kii_r - Kjj_r
                if eta_r >= -1e-12:
                    continue

                if yi_r != yj_r:
                    L = max(0.0, aj_r - ai_r)
                    H = min(C_vec[j_r], C_vec[i_r] + aj_r - ai_r)
                else:
                    L = max(0.0, ai_r + aj_r - C_vec[i_r])
                    H = min(C_vec[j_r], ai_r + aj_r)
                if L >= H:
                    continue

                aj_new = aj_r - yj_r * (Ei_r - Ej_r) / eta_r
                aj_new = min(max(aj_new, L), H)
                ai_new = ai_r + yi_r * yj_r * (aj_r - aj_new)
                ai_new = min(max(ai_new, 0.0), C_vec[i_r])
                if abs(ai_new - ai_r) < eps and abs(aj_new - aj_r) < eps:
                    continue

                dai = ai_new - ai_r
                daj = aj_new - aj_r
                b1 = b - Ei_r - yi_r * dai * Kii_r - yj_r * daj * Kij_r
                b2 = b - Ej_r - yi_r * dai * Kij_r - yj_r * daj * Kjj_r
                if 0 < ai_new < C_vec[i_r]:
                    b_new = b1
                elif 0 < aj_new < C_vec[j_r]:
                    b_new = b2
                else:
                    b_new = 0.5 * (b1 + b2)

                errors += yi_r * dai * row_r
                errors += yj_r * daj * provider.row(j_r)
                errors += b_new - b
                alpha[i_r] = ai_new
                alpha[j_r] = aj_new
                b = float(b_new)
                updated = True
                break

        if not updated:
            converged = False
            break

        n_iter = iteration
        if verbose and iteration % 1000 == 0:
            print(f"[SMO] iter={iteration:6d} kkt={kkt_violation:.6e} b={b:.6f}")

    n_iter = iteration
    clean_bounds()
    f = _decision_values(provider, alpha, y, b)
    errors = f - y
    kkt_violation = _kkt_violation(y, alpha, C_vec, f)

    return SMOResult(
        alpha=alpha,
        b=b,
        errors=errors,
        converged=converged,
        n_iter=n_iter,
        kkt_violation=float(kkt_violation),
    )
