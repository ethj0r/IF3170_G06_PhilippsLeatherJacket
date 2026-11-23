"""Modular model implementations package.

Add each new model in its own file (e.g., dtl.py, logreg.py, svm.py, nb.py)
and re-export it here so main.ipynb can import all 5 models cleanly.
"""

from .base import BaseModel
from .knn import DEFAULT_KNN_CONFIGS, KNN, get_knn_preprocessing_config

__all__ = [
    "BaseModel",
    "DEFAULT_KNN_CONFIGS",
    "KNN",
    "get_knn_preprocessing_config",
]
