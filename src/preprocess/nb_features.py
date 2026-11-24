import itertools

import numpy as np
import pandas as pd

BASE_NUMERICAL = ['lead', 'price', 'nights', 'arrival_date', 'booking_date']
EXTRA_NUMERICAL = ['price_per_guest', 'total_cost']
BASE_PAIRS = [
    'lead', 'year_month', 'requests', 'channel', 'price', 'room', 'weekend_nights', 'weekday_nights',
    'adults', 'children', 'meal', 'parking', 'returning', 'nights', 'month', 'weekday',
    'booking_date', 'arrival_date',
]
EXTRA_PAIRS = ['price_per_guest', 'total_cost', 'guests', 'week']
MISSING_COLUMNS = [
    'special_requests', 'nightly_rate_idr', 'days_before_arrival', 'num_children',
    'num_adults', 'meal_package', 'needs_parking', 'room_category',
]
FLAG_PAIRS = ['n_missing', 'special_requests_missing']
TRIPLE_BASE = [
    'lead', 'arrival_date', 'booking_date', 'channel', 'requests', 'price', 'room', 'weekend_nights',
    'weekday_nights', 'adults', 'meal', 'parking', 'children', 'year_month', 'weekday', 'returning',
]
QUAD_BASE = ['lead', 'arrival_date', 'channel', 'requests', 'price', 'room', 'weekend_nights', 'adults']
FINE_NUMERICAL = ['lead', 'price', 'arrival_date', 'booking_date', 'total_cost']
FINE_CATEGORICAL = ['channel', 'requests', 'room', 'weekend_nights', 'weekday_nights', 'adults']
MEAL = ['Room Only', 'Breakfast', 'Half Board', 'Full Board']
ROOM = [f'Type {c}' for c in 'ABCDEFG']
CHANNEL = ['Airline', 'Complimentary', 'Corporate', 'Offline Agent', 'Online Travel Agent']
NB_COLUMNS = [
    'meal', 'room', 'channel', 'parking', 'year', 'month', 'year_month', 'weekday', 'returning',
    'requests', 'adults', 'children', 'weekend_nights', 'weekday_nights', 'prior_cancel',
    'prior_booking', 'lead', 'price', 'nights', 'arrival_date', 'booking_date',
    'guests', 'week', 'price_per_guest', 'total_cost',
] + [f'{c}_missing' for c in MISSING_COLUMNS] + ['n_missing']


def prepare_nb_features(df, extra=False, flags=False):
    date_parts = df[['arrival_year', 'arrival_month', 'arrival_day']].rename(
        columns={'arrival_year': 'year', 'arrival_month': 'month', 'arrival_day': 'day'}
    )
    arrival = pd.to_datetime(date_parts, errors='coerce')

    X = pd.DataFrame(index=df.index)
    X['meal'] = df['meal_package'].map({v: i for i, v in enumerate(MEAL)})
    X['room'] = df['room_category'].map({v: i for i, v in enumerate(ROOM)})
    X['channel'] = df['booking_channel'].map({v: i for i, v in enumerate(CHANNEL)})
    X['parking'] = df['needs_parking']
    X['year'] = df['arrival_year']
    X['month'] = df['arrival_month']
    X['year_month'] = df['arrival_year'] * 100 + df['arrival_month']
    X['weekday'] = arrival.dt.dayofweek
    X['returning'] = df['is_returning_guest']
    X['requests'] = df['special_requests']
    X['adults'] = df['num_adults'].mask(df['num_adults'] == 0)
    X['children'] = df['num_children'].clip(upper=3)
    X['weekend_nights'] = df['weekend_nights'].clip(upper=4)
    X['weekday_nights'] = df['weekday_nights'].clip(upper=7)
    X['prior_cancel'] = (df['prior_cancellations'] > 0).astype(int)
    X['prior_booking'] = (df['prior_completed_bookings'] > 0).astype(int)
    X['lead'] = df['days_before_arrival']
    X['price'] = df['nightly_rate_idr']
    X['nights'] = df['weekend_nights'] + df['weekday_nights']
    X['arrival_date'] = (arrival - pd.Timestamp('2023-01-01')).dt.days
    X['booking_date'] = X['arrival_date'] - df['days_before_arrival']

    if extra:
        guests = X['adults'] + df['num_children']
        X['guests'] = guests.clip(upper=5)
        X['week'] = arrival.dt.isocalendar().week.astype(float)
        X['price_per_guest'] = X['price'] / guests.where(guests > 0)
        X['total_cost'] = X['price'] * X['nights']

    if flags:
        for c in MISSING_COLUMNS:
            X[f'{c}_missing'] = df[c].isna().astype(int)
        X['n_missing'] = df[MISSING_COLUMNS].isna().sum(axis=1).clip(upper=3)
    return X


def quantile_cuts(values, y, n_bins):
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.nanquantile(values, quantiles))


def row_entropy(counts):
    total = counts.sum(axis=1, keepdims=True)
    p = np.where(counts > 0, counts / np.maximum(total, 1), 1)
    return -np.sum(np.where(counts > 0, p * np.log2(p), 0), axis=1)


def mdl_split(v, t, cuts):
    """Recursive minimum description length (MDL) entropy discretization.

    Reference:
    - U. M. Fayyad and K. B. Irani (1993), Multi-Interval Discretization of
      Continuous-Valued Attributes for Classification Learning, IJCAI, pp. 1022-1027.
    """
    n = len(v)
    onehot = np.column_stack([t == 0, t == 1]).astype(float)
    total = onehot.sum(axis=0)
    boundary = np.nonzero(v[1:] != v[:-1])[0]
    if len(boundary) == 0 or np.count_nonzero(total) < 2:
        return
    left = np.cumsum(onehot, axis=0)[boundary]
    right = total - left
    n_left = left.sum(axis=1)
    weighted = (n_left * row_entropy(left) + (n - n_left) * row_entropy(right)) / n
    best = np.argmin(weighted)
    k = boundary[best]

    ent = row_entropy(total[None])[0]
    ent_left = row_entropy(left[best:best + 1])[0]
    ent_right = row_entropy(right[best:best + 1])[0]
    k_all = np.count_nonzero(total)
    k_left = np.count_nonzero(left[best])
    k_right = np.count_nonzero(right[best])
    gain = ent - weighted[best]
    delta = np.log2(3 ** k_all - 2) - (k_all * ent - k_left * ent_left - k_right * ent_right)
    if gain <= (np.log2(n - 1) + delta) / n:
        return

    cuts.append((v[k] + v[k + 1]) / 2)
    mdl_split(v[:k + 1], t[:k + 1], cuts)
    mdl_split(v[k + 1:], t[k + 1:], cuts)


def mdl_cuts(values, y, n_bins=None):
    known = ~np.isnan(values)
    order = np.argsort(values[known], kind='stable')
    cuts = []
    mdl_split(values[known][order], np.asarray(y)[known][order], cuts)
    return np.array(sorted(cuts))


class NBFeatures:
    def __init__(self, binning='quantile', n_bins=20, missing='skip', extra=False, flags=False, pairs=True):
        self.binning = binning
        self.n_bins = n_bins
        self.missing = missing
        self.extra = extra
        self.flags = flags
        self.pairs = pairs

    def fit(self, X, y):
        self.numerical_ = BASE_NUMERICAL + (EXTRA_NUMERICAL if self.extra else [])
        cut_fn = mdl_cuts if self.binning == 'mdl' else quantile_cuts
        self.cuts_ = {c: cut_fn(X[c].astype(float).values, y, self.n_bins) for c in self.numerical_}
        base = BASE_PAIRS + (EXTRA_PAIRS if self.extra else []) + (FLAG_PAIRS if self.flags else [])
        self.pair_list_ = list(itertools.combinations(base, 2)) if self.pairs else []
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.numerical_:
            values = X[c].astype(float)
            X[c] = np.where(values.isna(), np.nan, np.digitize(values, self.cuts_[c]))
        if self.missing == 'category':
            X = X.astype(object).where(X.notna(), 'missing')

        joined = {}
        for a, b in self.pair_list_:
            value = X[a].astype(str) + '|' + X[b].astype(str)
            joined[f'{a}*{b}'] = value.where(X[a].notna() & X[b].notna(), np.nan)
        return pd.concat([X, pd.DataFrame(joined, index=X.index)], axis=1)


class WideFeatures:
    def __init__(self, fine_bins=(10, 40), triples=TRIPLE_BASE, quads=QUAD_BASE, fine_crosses=False):
        self.fine_bins = fine_bins
        self.triples = triples
        self.quads = quads
        self.fine_crosses = fine_crosses

    def fit(self, X, y):
        self.base_ = NBFeatures(binning='mdl', missing='category', extra=True, flags=True, pairs=False).fit(X, y)
        self.fine_cuts_ = {
            (c, k): quantile_cuts(X[c].astype(float).values, y, k)
            for k in self.fine_bins for c in self.base_.numerical_
        }
        if self.fine_crosses:
            self.cross_cuts_ = {
                (c, k): quantile_cuts(X[c].astype(float).values, y, k)
                for c in FINE_NUMERICAL for k in (10, 20, 40)
            }
        return self

    def _fine_crosses(self, X):
        binned = {}
        for (c, k), cuts in self.cross_cuts_.items():
            values = X[c].astype(float)
            binned[(c, k)] = pd.Series(
                np.where(values.isna(), 'm', np.digitize(values, cuts).astype(str)), index=X.index
            )
        crosses = {}
        grid = [(c, k) for c in FINE_NUMERICAL[:4] for k in (20, 40)]
        for (a, ka), (b, kb) in itertools.combinations(grid, 2):
            if a != b:
                crosses[f'{a}{ka}x{b}{kb}'] = binned[(a, ka)] + '|' + binned[(b, kb)]
        for a, b in itertools.combinations(FINE_NUMERICAL, 2):
            for c in FINE_CATEGORICAL:
                crosses[f'{a}10x{b}10x{c}'] = binned[(a, 10)] + '|' + binned[(b, 10)] + '|' + X[c].astype(str)
        return pd.DataFrame(crosses, index=X.index)

    def transform(self, X):
        S = self.base_.transform(X)
        fine = {}
        for (c, k), cuts in self.fine_cuts_.items():
            values = X[c].astype(float)
            fine[f'{c}_q{k}'] = np.where(values.isna(), 'missing', np.digitize(values, cuts).astype(str))

        text = S.astype(str)
        crosses = {}
        for size, columns in [(2, list(S.columns)), (3, self.triples), (4, self.quads)]:
            for combo in itertools.combinations(columns, size):
                value = text[combo[0]]
                for c in combo[1:]:
                    value = value + '|' + text[c]
                crosses['*'.join(combo)] = value
        parts = [S, pd.DataFrame(fine, index=X.index), pd.DataFrame(crosses, index=X.index)]
        if self.fine_crosses:
            parts.append(self._fine_crosses(X))
        return pd.concat(parts, axis=1)
