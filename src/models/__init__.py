"""Modular model implementations package.

Add each new model in its own file (e.g., dtl.py, logreg.py, svm.py, nb.py)
and re-export it here so main.ipynb can import all 5 models cleanly.
"""

from .base import BaseModel
from .dtl import DEFAULT_DTL_CONFIGS, C45DecisionTree
from .knn import DEFAULT_KNN_CONFIGS, KNN, get_knn_preprocessing_config
from .naive_bayes import DEFAULT_NB_CONFIGS, NaiveBayes, WeightedNaiveBayes, WideNaiveBayes

__all__ = [
    "BaseModel",
    "C45DecisionTree",
    "DEFAULT_DTL_CONFIGS",
    "DEFAULT_KNN_CONFIGS",
    "DEFAULT_NB_CONFIGS",
    "KNN",
    "NaiveBayes",
    "WeightedNaiveBayes",
    "WideNaiveBayes",
    "get_knn_preprocessing_config",
]
