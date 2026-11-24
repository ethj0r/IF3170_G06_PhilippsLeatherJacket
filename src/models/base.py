"""Abstract base contract for all supervised classification models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BaseModel(ABC):
    """Base interface shared across all 5 model implementations."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseModel:
        """Fit model parameters or training index on training features and labels."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""

    def get_params(self) -> dict[str, Any]:
        """Return public hyperparameters stored on the model instance."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and not key.endswith("_")
        }

    def save(self, filepath: str | Path) -> Path:
        """Serialize the model instance to a .pkl file using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, filepath: str | Path) -> BaseModel:
        """Load a serialized model instance from a .pkl file."""
        loaded = joblib.load(Path(filepath))
        if not isinstance(loaded, cls):
            raise TypeError(f"Loaded object is {type(loaded).__name__}, expected {cls.__name__}.")
        return loaded
