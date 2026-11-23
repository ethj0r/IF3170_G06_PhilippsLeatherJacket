"""Stage 6: Univariate ANOVA F-value feature selection."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from .config import DEFAULT_TOP_K_FEATURES


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    test_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K_FEATURES,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame, SelectKBest]:
    """Rank features with SelectKBest(f_classif) and retain the top_k features for train and test."""
    k = min(top_k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)

    feature_summary = (
        pd.DataFrame({"Feature": X_train.columns, "Score": selector.scores_})
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )

    selected_features = feature_summary.head(top_k)["Feature"].tolist()
    X_train_final = X_train[selected_features].copy()
    X_test_final = test_df[selected_features].copy()

    if verbose:
        print("Features score (sorted):")
        print(feature_summary)
        print()

    return X_train_final, X_test_final, selected_features, feature_summary, selector
