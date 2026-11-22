import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from src.models.naive_bayes import NaiveBayes


def score(y_true, y_pred, metric):
    if metric == 'macro_f1':
        return f1_score(y_true, y_pred, average='macro')
    return accuracy_score(y_true, y_pred)


def make_folds(X, y, make_features, n_splits=5, seed=824):
    folds = []
    for a, b in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y):
        features = make_features().fit(X.iloc[a], y[a])
        folds.append((features.transform(X.iloc[a]), y[a], features.transform(X.iloc[b]), y[b]))
    return folds


def cv_score(folds, columns, metric, alpha=1.0):
    scores = []
    for X_a, y_a, X_b, y_b in folds:
        model = NaiveBayes(categorical_features=range(len(columns)), alpha=alpha)
        model.fit(X_a[columns].values, y_a)
        scores.append(score(y_b, model.predict(X_b[columns].values), metric))
    return float(np.mean(scores))


def forward_selection(folds, metric, max_features=10, floating=False, verbose=True):
    candidates = list(folds[0][0].columns)
    selected = []
    best = 0.0
    start = time.time()
    while len(selected) < max_features:
        results = [(cv_score(folds, selected + [c], metric), c) for c in candidates if c not in selected]
        value, feature = max(results)
        if value <= best:
            break
        selected.append(feature)
        best = value
        if verbose:
            print(f'  + {feature:30s} {metric} = {best:.5f}  ({time.time() - start:.0f} s)', flush=True)

        while floating and len(selected) > 2:
            results = [(cv_score(folds, [c for c in selected if c != r], metric), r) for r in selected[:-1]]
            value, removed = max(results)
            if value <= best:
                break
            selected.remove(removed)
            best = value
            if verbose:
                print(f'  - {removed:30s} {metric} = {best:.5f}  ({time.time() - start:.0f} s)', flush=True)
    return selected, best
