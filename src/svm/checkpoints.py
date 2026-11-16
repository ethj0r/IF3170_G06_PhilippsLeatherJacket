"""Submission checkpoints and best-score history."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = ROOT / "output" / "checkpoints"


def _now_utc() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ"), now.isoformat()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "submission"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def register_submission(
    submission_path: str | Path,
    *,
    name: str | None = None,
    public_score: float | None = None,
    holdout_accuracy: float | None = None,
    holdout_macro_f1: float | None = None,
    holdout_auc: float | None = None,
    model_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    notes: str | None = None,
    source_script: str | None = None,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> dict[str, Any]:
    """Register a submission and archive it if it sets a new best."""
    submission = _resolve(submission_path)
    if not submission.exists():
        raise FileNotFoundError(f"Submission not found: {submission}")
    df = pd.read_csv(submission)
    missing = {"id", "is_canceled"} - set(df.columns)
    if missing:
        raise ValueError(f"Submission is missing columns: {sorted(missing)}")

    checkpoint_dir = _resolve(checkpoint_dir)
    submissions_dir = checkpoint_dir / "submissions"
    public_dir = checkpoint_dir / "best_public"
    holdout_dir = checkpoint_dir / "best_holdout"
    models_dir = checkpoint_dir / "models"
    history_path = checkpoint_dir / "history.json"
    best_public_path = checkpoint_dir / "best_public.json"
    best_holdout_path = checkpoint_dir / "best_holdout.json"

    timestamp, iso_timestamp = _now_utc()
    name = _safe_name(name or submission.stem)
    submission_md5 = _md5(submission)
    row_count = int(len(df))
    positive_count = int(df["is_canceled"].sum())

    history = _load_json(
        history_path,
        {"records": [], "best_public": None, "best_holdout": None},
    )
    best_public = history.get("best_public")
    best_holdout = history.get("best_holdout")

    new_public_best = public_score is not None and (
        best_public is None or public_score > float(best_public["public_score"])
    )
    new_holdout_best = holdout_accuracy is not None and (
        best_holdout is None
        or holdout_accuracy > float(best_holdout["holdout_accuracy"])
    )

    record: dict[str, Any] = {
        "timestamp": iso_timestamp,
        "name": name,
        "submission_path": str(submission),
        "submission_md5": submission_md5,
        "rows": row_count,
        "positives": positive_count,
        "positive_rate": float(positive_count / row_count) if row_count else 0.0,
        "public_score": public_score,
        "holdout_accuracy": holdout_accuracy,
        "holdout_macro_f1": holdout_macro_f1,
        "holdout_auc": holdout_auc,
        "model_path": str(_resolve(model_path)) if model_path is not None else None,
        "model_md5": None,
        "config": config or {},
        "notes": notes,
        "source_script": source_script,
        "new_public_best": bool(new_public_best),
        "new_holdout_best": bool(new_holdout_best),
    }

    archived_submission = _copy(
        submission,
        submissions_dir / f"{timestamp}_{name}_{submission_md5[:8]}.csv",
    )
    record["archived_submission"] = str(archived_submission)

    model_copy = None
    if model_path is not None:
        model = _resolve(model_path)
        if not model.exists():
            raise FileNotFoundError(f"Model not found: {model}")
        record["model_md5"] = _md5(model)
        if new_public_best or new_holdout_best:
            model_copy = _copy(
                model,
                models_dir / f"{timestamp}_{name}_{record['model_md5'][:8]}.pkl",
            )
            record["archived_model"] = str(model_copy)

    if new_public_best:
        archive = _copy(
            submission,
            public_dir
            / f"{float(public_score):.6f}_{timestamp}_{name}.csv",
        )
        record["best_public_archive"] = str(archive)
        best_public = dict(record)
        history["best_public"] = best_public
        _atomic_json(best_public_path, best_public)

    if new_holdout_best:
        archive = _copy(
            submission,
            holdout_dir
            / f"{float(holdout_accuracy):.6f}_{timestamp}_{name}.csv",
        )
        record["best_holdout_archive"] = str(archive)
        best_holdout = dict(record)
        history["best_holdout"] = best_holdout
        _atomic_json(best_holdout_path, best_holdout)

    history.setdefault("records", []).append(record)
    _atomic_json(history_path, history)
    return record


def load_best_public(checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR):
    return _load_json(_resolve(checkpoint_dir) / "best_public.json", None)


def load_best_holdout(checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR):
    return _load_json(_resolve(checkpoint_dir) / "best_holdout.json", None)
