"""Abstract base contract for all supervised classification models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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
