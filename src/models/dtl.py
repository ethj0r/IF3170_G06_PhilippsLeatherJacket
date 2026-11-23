from numbers import Integral, Real

import numpy as np

from src.models.base import BaseModel


DEFAULT_DTL_CONFIGS = [
    {"max_depth": 16, "min_branch_weight": 7, "threshold": 0.48},
]


class C45DecisionTree(BaseModel):
    def __init__(
        self, categorical_features=None, *, max_depth=None, min_branch_weight=0, threshold=None
    ):
        if max_depth is not None and (
            not isinstance(max_depth, Integral) or max_depth < 0
        ):
            raise ValueError("max_depth must be a nonnegative integer or None.")
        if (
            not isinstance(min_branch_weight, Real)
            or not np.isfinite(min_branch_weight)
            or min_branch_weight < 0
        ):
            raise ValueError("min_branch_weight must be a finite nonnegative number.")
        if threshold is not None and (
            not isinstance(threshold, Real) or not 0 < threshold < 1
        ):
            raise ValueError("threshold must be between 0 and 1 or None.")
        self.threshold = threshold
        self.max_depth = max_depth
        self.min_branch_weight = min_branch_weight
        self.categorical_features = None
        if categorical_features is not None:
            self.categorical_features = tuple(categorical_features)
        self.tree_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=object)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError(
                "X must be a nonempty 2D input with one label per row in y."
            )
        if any(_is_missing(label) for label in y):
            raise ValueError("y cannot contain missing labels.")

        categorical = self.categorical_features
        if categorical is None:
            categorical = []
            for i, column in enumerate(X.T):
                for value in column:
                    if _is_missing(value):
                        continue
                    if not isinstance(value, Real) or isinstance(
                        value, (bool, np.bool_)
                    ):
                        categorical.append(i)
                        break
        elif any(
            not isinstance(i, Integral) or not 0 <= i < X.shape[1] for i in categorical
        ):
            raise ValueError("categorical_features must contain valid column indices.")

        classes = list(dict.fromkeys(y))
        class_index = {label: i for i, label in enumerate(classes)}
        if getattr(self, "threshold", None) is not None and len(classes) > 2:
            raise ValueError("A prediction threshold requires binary class labels.")
        labels = np.array([class_index[label] for label in y])
        X = _prepare_features(X, categorical)
        features = {i: i in categorical for i in range(X.shape[1])}

        self.tree_ = _build_tree(
            X,
            labels,
            np.ones(len(y)),
            features,
            len(classes),
            self.max_depth,
            self.min_branch_weight,
        )
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = X.shape[1]
        self._categorical_columns = categorical
        return self

    def predict_proba(self, X):
        if self.tree_ is None:
            raise RuntimeError("The model must be fitted before prediction.")
        X = np.asarray(X, dtype=object)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same columns as the training data.")

        X = _prepare_features(X, self._categorical_columns)
        probabilities = np.empty((len(X), len(self.classes_)))
        for i, row in enumerate(X):
            probabilities[i] = _predict_row(self.tree_, row)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        threshold = getattr(self, "threshold", None)
        if threshold is None or len(self.classes_) == 1:
            return self.classes_[probabilities.argmax(axis=1)]

        positive = 1 if 1 in self.classes_ else self.classes_[1]
        positive_index = list(self.classes_).index(positive)
        negative_index = 1 - positive_index
        is_positive = probabilities[:, positive_index] >= threshold
        return np.where(is_positive, positive, self.classes_[negative_index])


def _build_tree(X, y, weights, features, n_classes, depth_left, min_branch_weight):
    class_weights = np.bincount(y, weights=weights, minlength=n_classes)
    node = {"probabilities": class_weights / class_weights.sum(), "children": {}}
    if np.count_nonzero(class_weights) == 1 or depth_left == 0:
        return node

    split = _find_split(X, y, weights, features, n_classes, min_branch_weight)
    if split is None:
        return node

    feature, threshold = split
    node["feature"] = feature
    node["threshold"] = threshold
    column = X[:, feature]
    missing = _missing_rows(column)
    known_weight = weights[~missing].sum()

    if threshold is None:
        categories = dict.fromkeys(column[~missing])
        groups = {value: ~missing & (column == value) for value in categories}
        features = features.copy()
        del features[feature]
    else:
        groups = {"left": column <= threshold, "right": column > threshold}

    next_depth = None if depth_left is None else depth_left - 1
    for branch, rows in groups.items():
        fraction = weights[rows].sum() / known_weight
        selected = rows | missing
        child_weights = weights[selected]
        # Share the weight of each missing row across the branches.
        child_weights[missing[selected]] *= fraction
        child = _build_tree(
            X[selected],
            y[selected],
            child_weights,
            features,
            n_classes,
            next_depth,
            min_branch_weight,
        )
        node["children"][branch] = (child, fraction)
    return node


def _find_split(X, y, weights, features, n_classes, min_branch_weight):
    candidates = []
    for feature, is_categorical in features.items():
        column = X[:, feature]
        known = ~_missing_rows(column)
        values = column[known]
        if len(values) < 2:
            continue
        known_labels = y[known]
        known_weights = weights[known]
        missing_weight = weights[~known].sum()
        if known_weights.sum() < 2 * min_branch_weight - 1e-10:
            continue

        if is_categorical:
            categories = list(dict.fromkeys(values))
            if len(categories) < 2:
                continue
            thresholds = [None]
        else:
            values = values.astype(float)
            thresholds = np.unique(values)[:-1]

        best = None
        best_ratio = -1.0
        for threshold in thresholds:
            if threshold is None:
                groups = [values == category for category in categories]
            else:
                left = values <= threshold
                groups = [left, ~left]

            counts = []
            for rows in groups:
                count = np.bincount(
                    known_labels[rows], weights=known_weights[rows], minlength=n_classes
                )
                counts.append(count)
            counts = np.array(counts)
            # A split needs at least two branches with enough known weight.
            if min_branch_weight:
                large_branches = counts.sum(axis=1) >= min_branch_weight - 1e-10
                if np.count_nonzero(large_branches) < 2:
                    continue
            gain, ratio = _gain_ratio(counts, missing_weight)
            if ratio > best_ratio + 1e-12:
                best_ratio = ratio
                best = (feature, threshold, gain, ratio)

        if best is not None:
            candidates.append(best)

    if not candidates:
        return None

    # Ignore splits with below-average gain before comparing gain ratios.
    average_gain = np.mean([gain for _, _, gain, _ in candidates])
    best_split = None
    best_ratio = -1.0
    for feature, threshold, gain, ratio in candidates:
        if gain >= average_gain - 1e-12 and ratio > best_ratio:
            best_ratio = ratio
            best_split = (feature, threshold) if gain > 1e-12 else None
    return best_split


def _gain_ratio(counts, missing_weight):
    branch_weights = counts.sum(axis=1)
    known_weight = branch_weights.sum()
    total_weight = known_weight + missing_weight

    before_split = _entropy(counts.sum(axis=0))
    after_split = 0.0
    for count, weight in zip(counts, branch_weights):
        after_split += (weight / known_weight) * _entropy(count)
    gain = (known_weight / total_weight) * (before_split - after_split)

    split_info = _entropy(np.append(branch_weights, missing_weight))
    if split_info == 0:
        return gain, 0.0
    return gain, gain / split_info


def _entropy(weights):
    positive = weights[weights > 0]
    if len(positive) == 0:
        return 0.0
    probabilities = positive / positive.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _predict_row(node, row):
    children = node["children"]
    if not children:
        return node["probabilities"]

    value = row[node["feature"]]
    threshold = node["threshold"]
    if not _is_missing(value):
        branch = value
        if threshold is not None:
            branch = "left" if value <= threshold else "right"
        if branch in children:
            child, _ = children[branch]
            return _predict_row(child, row)

    # With no matching branch, average the predictions using the branch weights.
    probabilities = np.zeros_like(node["probabilities"])
    for child, fraction in children.values():
        probabilities += fraction * _predict_row(child, row)
    return probabilities


def _prepare_features(X, categorical):
    X = X.copy()
    for i in range(X.shape[1]):
        if i not in categorical:
            column = X[:, i].astype(float)
            if np.isinf(column).any():
                raise ValueError(f"Numeric feature {i} cannot contain infinity.")
            X[:, i] = column
    return X


def _missing_rows(column):
    return np.array([_is_missing(value) for value in column], dtype=bool)


def _is_missing(value):
    return value is None or (isinstance(value, Real) and value != value)
