import numpy as np
import pandas as pd

from glassbox import config
from glassbox.data import apply_imputer, fit_imputer, load_dataset


def test_split_sizes_and_stratification():
    d = load_dataset()
    n = len(d.y_train) + len(d.y_val) + len(d.y_test)
    assert n == 768
    assert abs(len(d.y_test) / n - config.TEST_SIZE) < 0.01
    assert abs(len(d.y_val) / n - config.VAL_SIZE) < 0.01
    rates = [y.mean() for y in (d.y_train, d.y_val, d.y_test)]
    assert max(rates) - min(rates) < 0.02          # same diabetic share in every split


def test_no_impossible_zeros_after_imputation():
    d = load_dataset()
    cols = [d.feature_names.index(c) for c in config.ZERO_MEANS_MISSING]
    for X in (d.X_train, d.X_val, d.X_test):
        assert not np.any(X[:, cols] == 0)


def test_imputer_uses_only_nonzero_training_values():
    X = pd.DataFrame({c: [0.0, 10.0, 20.0, 30.0] for c in config.ZERO_MEANS_MISSING})
    medians = fit_imputer(X)
    assert all(m == 20.0 for m in medians.values())
    filled = apply_imputer(X, medians)
    assert (filled.iloc[0] == 20.0).all()
    assert X.iloc[0, 0] == 0.0                     # original left untouched
