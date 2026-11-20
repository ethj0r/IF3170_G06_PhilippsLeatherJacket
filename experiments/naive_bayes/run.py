import argparse
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.nb_features import NBFeatures, prepare_nb_features as prepare
from experiments.naive_bayes.search import cv_score, forward_selection, make_folds
from src.naive_bayes import NaiveBayes

RANDOM_STATE = 824
RESULTS_PATH = Path(__file__).parent / 'results.csv'

CONFIGS = {
    'baseline_quantile20': dict(binning='quantile', n_bins=20),
    'quantile10': dict(binning='quantile', n_bins=10),
    'mdl': dict(binning='mdl'),
    'mdl_missing_category': dict(binning='mdl', missing='category'),
    'mdl_extra': dict(binning='mdl', extra=True),
    'mdl_extra_floating': dict(binning='mdl', extra=True, floating=True),
    'mdl_flags': dict(binning='mdl', flags=True),
    'mdl_missing_category_flags': dict(binning='mdl', missing='category', flags=True),
}


def run(name, config, metric, train_df):
    config = dict(config)
    floating = config.pop('floating', False)
    extra = config.get('extra', False)
    flags = config.get('flags', False)

    X = prepare(train_df, extra=extra, flags=flags)
    y = train_df['is_canceled'].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    start = time.time()
    folds = make_folds(X_train, y_train, lambda: NBFeatures(**config))
    columns, cv_best = forward_selection(folds, metric, floating=floating)
    alphas = {a: cv_score(folds, columns, metric, a) for a in [0.5, 1, 2]}
    alpha = max(alphas, key=alphas.get)

    features = NBFeatures(**config).fit(X_train, y_train)
    model = NaiveBayes(categorical_features=range(len(columns)), alpha=alpha)
    model.fit(features.transform(X_train)[columns].values, y_train)
    pred = model.predict(features.transform(X_val)[columns].values)

    result = {
        'name': name,
        'metric': metric,
        'cv': round(alphas[alpha], 5),
        'holdout_accuracy': round(accuracy_score(y_val, pred), 5),
        'holdout_macro_f1': round(f1_score(y_val, pred, average='macro'), 5),
        'alpha': alpha,
        'n_features': len(columns),
        'features': ' '.join(columns),
        'seconds': round(time.time() - start),
    }
    return result, columns, alpha


def submit(name, config, columns, alpha, train_df, test_df):
    config = dict(config)
    config.pop('floating', None)
    extra = config.get('extra', False)
    flags = config.get('flags', False)

    X = prepare(train_df, extra=extra, flags=flags)
    y = train_df['is_canceled'].values
    features = NBFeatures(**config).fit(X, y)
    model = NaiveBayes(categorical_features=range(len(columns)), alpha=alpha)
    model.fit(features.transform(X)[columns].values, y)

    X_test = features.transform(prepare(test_df, extra=extra, flags=flags))[columns]
    pred = model.predict(X_test.values).astype(int)
    submission = pd.DataFrame({'id': test_df['id'], 'is_canceled': pred})
    path = ROOT / 'output' / f'submission_nb_{name}.csv'
    path.parent.mkdir(exist_ok=True)
    submission.to_csv(path, index=False)
    print(f'  submission -> {path.relative_to(ROOT)}  {submission["is_canceled"].value_counts().to_dict()}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='accuracy', choices=['accuracy', 'macro_f1'])
    parser.add_argument('--only', nargs='*', default=list(CONFIGS))
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()

    train_df = pd.read_csv(ROOT / 'data' / 'train.csv')
    test_df = pd.read_csv(ROOT / 'data' / 'test.csv')
    for name in args.only:
        print(f'== {name} ({args.metric})', flush=True)
        result, columns, alpha = run(name, CONFIGS[name], args.metric, train_df)
        if args.submit:
            submit(name, CONFIGS[name], columns, alpha, train_df, test_df)
        print(f"  cv={result['cv']}  holdout acc={result['holdout_accuracy']}  "
              f"macro_f1={result['holdout_macro_f1']}  alpha={result['alpha']}", flush=True)
        row = pd.DataFrame([result])
        row.to_csv(RESULTS_PATH, mode='a', header=not RESULTS_PATH.exists(), index=False)


if __name__ == '__main__':
    main()
