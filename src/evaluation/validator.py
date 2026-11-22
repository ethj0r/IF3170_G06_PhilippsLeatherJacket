"""Shared train/validation splitting, StratifiedKFold CV, and model comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.models.base import BaseModel


@dataclass
class SplitData:
    """Train/validation split arrays alongside original-vs-synthetic row masks."""

    X_tr: np.ndarray
    X_val: np.ndarray
    y_tr: np.ndarray
    y_val: np.ndarray
    orig_tr: np.ndarray
    orig_val: np.ndarray


def verify_smote_order(
    X_train_encoded: pd.DataFrame,
    X_train_balanced: pd.DataFrame,
    verbose: bool = True,
) -> bool:
    """Verify SMOTE keeps original rows at the front before synthetic samples."""
    n_orig = len(X_train_encoded)
    a = X_train_balanced.iloc[:n_orig].to_numpy(dtype=float)
    b = X_train_encoded.to_numpy(dtype=float)
    is_valid = bool(np.allclose(a, b))
    if verbose:
        print(f"Asumsi urutan SMOTE benar: {is_valid}")
    return is_valid


def split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    is_orig: np.ndarray | None = None,
    test_size: float = 0.15,
    random_state: int = 824,
    verbose: bool = True,
) -> SplitData:
    """Split training data while stratifying over label and original-vs-synthetic flag."""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    is_orig_arr = np.ones(len(y_arr), dtype=bool) if is_orig is None else np.asarray(is_orig, dtype=bool)

    if not is_orig_arr.all():
        strat_key = y_arr.astype(int) * 2 + is_orig_arr.astype(int)
    else:
        strat_key = y_arr.astype(int)

    X_tr, X_val, y_tr, y_val, orig_tr, orig_val = train_test_split(
        X_arr,
        y_arr,
        is_orig_arr,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key,
    )

    if verbose:
        print(f"Ukuran data train: {X_tr.shape}, Ukuran data validation: {X_val.shape}")
        print(
            f"Validation: {int(orig_val.sum())} baris asli, "
            f"{int((~orig_val).sum())} baris sintetis (tidak dinilai)\n"
        )

    return SplitData(
        X_tr=np.asarray(X_tr),
        X_val=np.asarray(X_val),
        y_tr=np.asarray(y_tr),
        y_val=np.asarray(y_val),
        orig_tr=np.asarray(orig_tr, dtype=bool),
        orig_val=np.asarray(orig_val, dtype=bool),
    )


def cross_validate_configs(
    model_cls: type[BaseModel],
    configs: list[dict[str, Any]],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    k_folds: int = 5,
    random_state: int = 824,
    model_name: str | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate hyperparameter configs via StratifiedKFold and return the best config by Macro F1."""
    if not configs:
        raise ValueError("configs must contain at least one hyperparameter dictionary")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    display_name = model_name or model_cls.__name__
    records: list[dict[str, Any]] = []
    best_f1_score = -1.0
    best_config = configs[0]

    for cfg in configs:
        fold_accuracies: list[float] = []
        fold_f1_scores: list[float] = []

        for train_idx, val_idx in skf.split(X_tr, y_tr):
            X_fold_train, X_fold_val = X_tr[train_idx], X_tr[val_idx]
            y_fold_train, y_fold_val = y_tr[train_idx], y_tr[val_idx]

            model = model_cls(**cfg)
            model.fit(X_fold_train, y_fold_train)
            y_pred_fold = model.predict(X_fold_val)

            fold_accuracies.append(float(accuracy_score(y_fold_val, y_pred_fold)))
            fold_f1_scores.append(float(f1_score(y_fold_val, y_pred_fold, average="macro")))

        mean_acc = float(np.mean(fold_accuracies))
        mean_f1 = float(np.mean(fold_f1_scores))
        std_f1 = float(np.std(fold_f1_scores))

        records.append(
            {
                "model": display_name,
                "config": cfg,
                "mean_accuracy": mean_acc,
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
            }
        )

        if verbose:
            cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items())
            print(
                f"[{display_name} | {cfg_str}] -> "
                f"Mean Acc: {mean_acc:.5f} | Mean Macro F1: {mean_f1:.5f} (±{std_f1:.5f})"
            )

        if mean_f1 > best_f1_score:
            best_f1_score = mean_f1
            best_config = cfg

    return best_config, pd.DataFrame(records)


def train_and_evaluate(
    model: BaseModel,
    split_data: SplitData,
    model_name: str = "Model",
    config: dict[str, Any] | None = None,
    X_test: np.ndarray | None = None,
    plot_cm: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train model on split_data.X_tr and evaluate strictly on original validation rows."""
    model.fit(split_data.X_tr, split_data.y_tr)
    val_preds = model.predict(split_data.X_val)

    mask = split_data.orig_val
    y_true_orig = split_data.y_val[mask]
    y_pred_orig = val_preds[mask]

    acc = float(accuracy_score(y_true_orig, y_pred_orig))
    macro_f1 = float(f1_score(y_true_orig, y_pred_orig, average="macro"))
    report = classification_report(y_true_orig, y_pred_orig)
    cm = confusion_matrix(y_true_orig, y_pred_orig)

    if verbose:
        print(f"\n=== Evaluation: {model_name} ===")
        print(f"Accuracy (original val): {acc:.5f} | Macro F1: {macro_f1:.5f}")
        print(report)

    if plot_cm:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        cfg_label = f": {config}" if config else ""
        plt.title(f"Confusion Matrix ({model_name}{cfg_label})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    return {
        "model_name": model_name,
        "model": model,
        "config": config if config is not None else model.get_params(),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": report,
        "confusion_matrix": cm,
        "val_preds": val_preds,
        "X_test": X_test,
    }


def compare_models(evaluation_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a sorted comparison table of evaluated models ordered by Macro F1."""
    rows = [
        {
            "Model": item["model_name"],
            "Accuracy": item["accuracy"],
            "Macro F1": item["macro_f1"],
            "Config": item.get("config"),
        }
        for item in evaluation_results
    ]
    return pd.DataFrame(rows).sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
