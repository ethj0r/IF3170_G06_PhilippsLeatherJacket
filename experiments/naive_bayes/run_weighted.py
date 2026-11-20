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
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.preprocess.nb_features import QUAD_BASE, TRIPLE_BASE, WideFeatures, prepare_nb_features as prepare
from src.models.naive_bayes import WeightedNaiveBayes

RANDOM_STATE = 824
RESULTS_PATH = Path(__file__).parent / 'results.csv'


def fit_predict(X_a, y_a, X_b, args):
    features = WideFeatures(
        triples=TRIPLE_BASE[:args.triples], quads=QUAD_BASE[:args.quads], fine_crosses=args.fine_crosses
    ).fit(X_a, y_a)
    A = features.transform(X_a)
    model = WeightedNaiveBayes(
        categorical_features=range(A.shape[1]), alpha=args.alpha, l2=args.l2,
        n_folds=args.inner_folds, random_state=RANDOM_STATE,
    )
    model.fit(A.values, y_a)
    return model.predict(features.transform(X_b).values), A.shape[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--l2', type=float, default=10.0)
    parser.add_argument('--inner-folds', type=int, default=5)
    parser.add_argument('--triples', type=int, default=16)
    parser.add_argument('--quads', type=int, default=8)
    parser.add_argument('--fine-crosses', action='store_true')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    name = f'wnb_wide_a{args.alpha:g}_l{args.l2:g}'
    if (args.inner_folds, args.triples, args.quads) != (5, 16, 8):
        name += f'_f{args.inner_folds}_t{args.triples}_q{args.quads}'
    if args.fine_crosses:
        name += '_fine'

    train_df = pd.read_csv(ROOT / 'data' / 'train.csv')
    X = prepare(train_df, extra=True, flags=True)
    y = train_df['is_canceled'].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    start = time.time()

    cv = np.nan
    if args.cv:
        scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for a, b in skf.split(X_train, y_train):
            pred, _ = fit_predict(X_train.iloc[a], y_train[a], X_train.iloc[b], args)
            scores.append(accuracy_score(y_train[b], pred))
            print(f'  fold accuracy = {scores[-1]:.5f}  ({time.time() - start:.0f} s)', flush=True)
        cv = round(float(np.mean(scores)), 5)

    pred, n_features = fit_predict(X_train, y_train, X_val, args)
    result = {
        'name': name,
        'metric': 'accuracy',
        'cv': cv,
        'holdout_accuracy': round(accuracy_score(y_val, pred), 5),
        'holdout_macro_f1': round(f1_score(y_val, pred, average='macro'), 5),
        'alpha': args.alpha,
        'n_features': n_features,
        'features': f'wide l2={args.l2:g} inner_folds={args.inner_folds} triples={args.triples} quads={args.quads} fine={args.fine_crosses}',
        'seconds': round(time.time() - start),
    }
    print(f"== {name}: cv={cv}  holdout acc={result['holdout_accuracy']}  macro_f1={result['holdout_macro_f1']}")
    pd.DataFrame([result]).to_csv(RESULTS_PATH, mode='a', header=not RESULTS_PATH.exists(), index=False)

    if args.submit:
        test_df = pd.read_csv(ROOT / 'data' / 'test.csv')
        pred, _ = fit_predict(X, y, prepare(test_df, extra=True, flags=True), args)
        submission = pd.DataFrame({'id': test_df['id'], 'is_canceled': pred.astype(int)})
        path = ROOT / 'output' / f'submission_{name}.csv'
        submission.to_csv(path, index=False)
        print(f'  submission -> {path.relative_to(ROOT)}  {submission["is_canceled"].value_counts().to_dict()}')


if __name__ == '__main__':
    main()
