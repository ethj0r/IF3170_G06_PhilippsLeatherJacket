"""Training, evaluation, serialization, and comparison utilities."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .data import HotelData
from .model import SVM


def stratified_subset_indices(
    y: np.ndarray,
    is_orig: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> np.ndarray:
    """Return a stratified subset of row indices."""
    n = len(y)
    if max_samples is None or max_samples >= n:
        return np.arange(n)

    keys = np.asarray(y).astype(int) * 2 + np.asarray(is_orig).astype(int)
    indices = np.arange(n)
    subset, _ = train_test_split(
        indices,
        train_size=int(max_samples),
        random_state=random_state,
        stratify=keys,
    )
    return np.sort(subset)


def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int,
) -> dict[str, float | str]:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_1": float(
            precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        ),
        "recall_1": float(
            recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        ),
        "f1_1": float(
            f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": json.dumps(cm.tolist()),
    }


def evaluate_model(
    model_type: str,
    kernel: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    orig_val: np.ndarray,
    C: float,
    gamma_requested: str | float,
    degree: int,
    coef0: float,
) -> dict:
    """Fit a model, measure timings, and evaluate on original validation rows."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = np.asarray(model.predict(X_val))
    predict_time = time.perf_counter() - start

    y_score = None
    if hasattr(model, "decision_function"):
        y_score = np.asarray(model.decision_function(X_val))

    mask = np.asarray(orig_val, dtype=bool)
    y_true_eval = np.asarray(y_val)[mask]
    y_pred_eval = y_pred[mask]
    pos_label = int(model.classes_[1])
    metrics = _binary_metrics(y_true_eval, y_pred_eval, pos_label=pos_label)

    if hasattr(model, "support_vectors_"):
        n_support = int(model.support_vectors_.shape[0])
    else:
        n_support = int(np.sum(model.n_support_))

    if model_type == "scratch":
        converged = bool(model.converged_)
        n_iter = int(model.n_iter_)
        kkt = float(model.kkt_violation_)
        gamma_resolved = float(model.gamma_)
    else:
        converged = bool(getattr(model, "fit_status_", 0) == 0)
        n_iter_raw = np.asarray(getattr(model, "n_iter_", -1)).ravel()
        n_iter = int(n_iter_raw[0]) if n_iter_raw.size else -1
        kkt = float("nan")
        gamma_resolved = float(getattr(model, "_gamma", 0.0))

    row = {
        "model_type": model_type,
        "kernel": kernel,
        "C": float(C),
        "gamma_requested": str(gamma_requested),
        "gamma_resolved": gamma_resolved,
        "degree": int(degree),
        "coef0": float(coef0),
        "n_train": int(X_train.shape[0]),
        "n_val_original": int(mask.sum()),
        "train_time_s": float(train_time),
        "predict_time_s": float(predict_time),
        "n_support": n_support,
        "converged": converged,
        "n_iter": n_iter,
        "kkt_violation": kkt,
        **metrics,
    }
    return {
        "row": row,
        "model": model,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def make_scratch_model(
    kernel: str,
    C: float,
    degree: int,
    gamma: str | float,
    coef0: float,
    tol: float,
    max_iter: int,
    random_state: int,
    cache_size_mb: float,
    verbose: bool,
) -> SVM:
    return SVM(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        cache_size_mb=cache_size_mb,
        verbose=verbose,
    )


def make_sklearn_model(
    kernel: str,
    C: float,
    degree: int,
    gamma: str | float,
    coef0: float,
) -> SVC:
    common = {"C": C, "cache_size": 500}
    if kernel == "linear":
        return SVC(kernel="linear", **common)
    if kernel == "poly":
        return SVC(
            kernel="poly",
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            **common,
        )
    if kernel == "rbf":
        return SVC(kernel="rbf", gamma=gamma, **common)
    raise ValueError(f"Unknown kernel {kernel!r}")


def print_row(row: dict) -> None:
    print(
        f"  {row['model_type']:7s} {row['kernel']:6s} "
        f"acc={row['accuracy']:.4f} f1(1)={row['f1_1']:.4f} "
        f"macroF1={row['macro_f1']:.4f} SV={row['n_support']:5d} "
        f"fit={row['train_time_s']:7.2f}s pred={row['predict_time_s']:.4f}s "
        f"conv={row['converged']}"
    )


def run_comparison(
    data: HotelData,
    kernels: list[str],
    max_samples: int | None,
    C: float,
    degree: int,
    gamma: str | float,
    coef0: float,
    tol: float,
    max_iter: int,
    random_state: int,
    cache_size_mb: float,
    verbose: bool = False,
) -> dict:
    """Train scratch and sklearn SVMs on the same subset and compare them."""
    indices = stratified_subset_indices(
        data.y_tr, data.orig_tr, max_samples=max_samples, random_state=random_state
    )
    X_sub = data.X_tr[indices]
    y_sub = data.y_tr[indices]
    orig_sub = data.orig_tr[indices]

    print(
        f"Training subset: {X_sub.shape[0]} rows, "
        f"class counts={dict(zip(*np.unique(y_sub, return_counts=True)))}, "
        f"original={int(orig_sub.sum())}, synthetic={int((~orig_sub).sum())}"
    )
    print(
        f"Validation: {int(data.orig_val.sum())} original rows, "
        f"{int((~data.orig_val).sum())} synthetic rows excluded from scoring"
    )

    rows: list[dict] = []
    models: dict[str, dict[str, object]] = {"scratch": {}, "sklearn": {}}
    predictions: dict[str, dict[str, dict]] = {"scratch": {}, "sklearn": {}}

    for kernel in kernels:
        print(f"\nKernel: {kernel}")
        scratch_model = make_scratch_model(
            kernel=kernel,
            C=C,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            cache_size_mb=cache_size_mb,
            verbose=verbose,
        )
        scratch_result = evaluate_model(
            model_type="scratch",
            kernel=kernel,
            model=scratch_model,
            X_train=X_sub,
            y_train=y_sub,
            X_val=data.X_val,
            y_val=data.y_val,
            orig_val=data.orig_val,
            C=C,
            gamma_requested=gamma,
            degree=degree,
            coef0=coef0,
        )
        print_row(scratch_result["row"])

        sklearn_model = make_sklearn_model(
            kernel=kernel,
            C=C,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
        )
        sklearn_result = evaluate_model(
            model_type="sklearn",
            kernel=kernel,
            model=sklearn_model,
            X_train=X_sub,
            y_train=y_sub,
            X_val=data.X_val,
            y_val=data.y_val,
            orig_val=data.orig_val,
            C=C,
            gamma_requested=gamma,
            degree=degree,
            coef0=coef0,
        )
        print_row(sklearn_result["row"])

        mask = data.orig_val
        s_scores = scratch_result["y_score"][mask]
        k_scores = sklearn_result["y_score"][mask]
        agreement = float(np.mean(np.sign(s_scores) == np.sign(k_scores)))
        if np.std(s_scores) > 0 and np.std(k_scores) > 0:
            correlation = float(np.corrcoef(s_scores, k_scores)[0, 1])
        else:
            correlation = float("nan")
        print(
            f"  comparison: sign_agreement={agreement:.4f} "
            f"score_correlation={correlation:.4f}"
        )

        for result in (scratch_result, sklearn_result):
            result["row"]["sign_agreement_vs_other"] = agreement
            result["row"]["score_correlation_vs_other"] = correlation

        rows.append(scratch_result["row"])
        rows.append(sklearn_result["row"])
        models["scratch"][kernel] = scratch_model
        models["sklearn"][kernel] = sklearn_model
        predictions["scratch"][kernel] = {
            "y_pred": scratch_result["y_pred"],
            "y_score": scratch_result["y_score"],
        }
        predictions["sklearn"][kernel] = {
            "y_pred": sklearn_result["y_pred"],
            "y_score": sklearn_result["y_score"],
        }

    return {
        "rows": rows,
        "models": models,
        "predictions": predictions,
        "subset_indices": indices,
        "X_sub": X_sub,
        "y_sub": y_sub,
        "orig_sub": orig_sub,
    }


def save_models(models: dict[str, dict[str, object]], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for model_type, model_dict in models.items():
        for kernel, model in model_dict.items():
            path = output_dir / f"svm_{model_type}_{kernel}.pkl"
            joblib.dump(model, path)
            paths[f"{model_type}_{kernel}"] = path
    return paths


def check_serialization(model, X: np.ndarray) -> bool:
    """Verify that a joblib round-trip preserves predictions exactly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.pkl"
        joblib.dump(model, path)
        loaded = joblib.load(path)
        before = np.asarray(model.predict(X))
        after = np.asarray(loaded.predict(X))
        return bool(np.array_equal(before, after))


def save_metrics(
    rows: list[dict],
    subset_indices: np.ndarray,
    artefacts_dir: str | Path,
    extra: dict | None = None,
) -> tuple[Path, Path]:
    artefacts_dir = Path(artefacts_dir)
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(rows)
    metrics_path = artefacts_dir / "svm_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    timing_payload = {
        "n_subset": int(len(subset_indices)),
        "models": [
            {
                "model_type": row["model_type"],
                "kernel": row["kernel"],
                "n_train": row["n_train"],
                "train_time_s": row["train_time_s"],
                "predict_time_s": row["predict_time_s"],
                "n_support": row["n_support"],
                "converged": row["converged"],
                "n_iter": row["n_iter"],
            }
            for row in rows
        ],
    }
    if extra:
        timing_payload.update(extra)
    timing_path = artefacts_dir / "svm_timing.json"
    timing_path.write_text(json.dumps(timing_payload, indent=2), encoding="utf-8")
    return metrics_path, timing_path


def best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str = "accuracy",
) -> tuple[float, float]:
    """Find the decision threshold that maximises accuracy or macro F1."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    best_threshold = 0.0
    best_value = -np.inf
    for threshold in np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 241))):
        predictions = (scores > threshold).astype(int)
        if metric == "accuracy":
            value = float(np.mean(predictions == y_true))
        elif metric == "macro_f1":
            value = float(f1_score(y_true, predictions, average="macro", zero_division=0))
        else:
            raise ValueError(f"Unknown threshold metric {metric!r}")
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold, best_value


def save_submission(
    model,
    X_test: np.ndarray,
    test_ids: np.ndarray,
    output_path: str | Path,
    sample_submission_path: str | Path = "data/sample_submission.csv",
    threshold: float | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if threshold is None:
        predictions = np.asarray(model.predict(X_test)).astype(int)
    else:
        predictions = (np.asarray(model.decision_function(X_test)) > threshold).astype(int)
    submission = pd.DataFrame({"id": test_ids, "is_canceled": predictions})

    sample = pd.read_csv(sample_submission_path)
    assert len(submission) == len(sample), "Submission row count does not match sample"
    assert list(submission.columns) == list(sample.columns), "Submission columns do not match"
    assert submission["is_canceled"].isnull().sum() == 0, "Submission contains NaN"
    submission.to_csv(output_path, index=False)
    return output_path


def best_scratch_kernel(rows: list[dict]) -> str:
    scratch_rows = [row for row in rows if row["model_type"] == "scratch"]
    if not scratch_rows:
        raise ValueError("No scratch rows available")
    best = max(scratch_rows, key=lambda row: (row["macro_f1"], row["f1_1"]))
    return str(best["kernel"])
