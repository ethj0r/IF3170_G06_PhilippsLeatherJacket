"""Visualisation helpers for the SVM bonus figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from .model import SVM


def _stratified_points(
    X: np.ndarray,
    y: np.ndarray,
    max_points: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_points:
        return X, y
    indices, _ = train_test_split(
        np.arange(X.shape[0]),
        train_size=int(max_points),
        random_state=random_state,
        stratify=y,
    )
    indices = np.sort(indices)
    return X[indices], y[indices]


def plot_2d_decision_boundary(
    X_2d: np.ndarray,
    y: np.ndarray,
    kernel: str,
    C: float,
    degree: int,
    gamma: str | float,
    coef0: float,
    output_path: str | Path,
    random_state: int = 824,
    max_points: int = 1500,
    tol: float = 1e-3,
    max_iter: int = 200,
) -> SVM:
    """Train a 2D illustration SVM and plot boundary, margins, and SVs.

    The model is fitted on the first two PCA components.  This is a separate
    2D illustration; it is not the full 9-dimensional model.
    """
    X_2d = np.asarray(X_2d, dtype=np.float64)[:, :2]
    y = np.asarray(y).ravel()
    X_plot, y_plot = _stratified_points(X_2d, y, max_points, random_state)

    model = SVM(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_plot, y_plot)

    # Use robust percentile limits so a few outliers do not compress the
    # central region where the boundary and margins are visible.
    x_min, x_max = np.percentile(X_plot[:, 0], [1.0, 99.0])
    y_min, y_max = np.percentile(X_plot[:, 1], [1.0, 99.0])
    x_pad = 0.5 * (x_max - x_min + 1e-9)
    y_pad = 0.5 * (y_max - y_min + 1e-9)
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 400),
        np.linspace(y_min - y_pad, y_max + y_pad, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.contourf(
        xx,
        yy,
        scores,
        levels=[-np.inf, -1.0, 1.0, np.inf],
        colors=["#f4c7c3", "#eeeeee", "#c7d7f4"],
        alpha=0.55,
    )
    ax.contour(
        xx,
        yy,
        scores,
        levels=[-1.0, 0.0, 1.0],
        colors=["#b22222", "#111111", "#1f4e9c"],
        linestyles=["--", "-", "--"],
        linewidths=[1.2, 2.0, 1.2],
    )

    for label, color, marker in [(0, "#1f4e9c", "o"), (1, "#b22222", "s")]:
        mask = y_plot == label
        ax.scatter(
            X_plot[mask, 0],
            X_plot[mask, 1],
            c=color,
            marker=marker,
            s=24,
            alpha=0.65,
            label=f"Class {label}",
            edgecolors="none",
        )

    if model.support_vectors_.size:
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            label="Support vectors",
        )

    gamma_label = f"{model.gamma_:.4g}" if kernel != "linear" else "n/a"
    ax.set_title(
        f"SVM {kernel} kernel (2D PCA illustration)\n"
        f"C={C}, gamma={gamma_label}, degree={degree}, coef0={coef0}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.01,
        0.01,
        "Decision boundary: f=0; dashed: margins f=±1",
        transform=ax.transAxes,
        fontsize=7,
        color="#333333",
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return model
