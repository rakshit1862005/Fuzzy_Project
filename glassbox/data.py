"""Loading, cleaning and splitting the Pima Indians Diabetes dataset."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


@dataclass
class Dataset:
    """Train / validation / test splits after cleaning, as NumPy arrays.

    Values stay in their original units (mg/dL, years, ...) so that fuzzy
    rules can later be read in units a doctor understands.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    medians: dict


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(config.FEATURES + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def fit_imputer(X_train: pd.DataFrame) -> dict:
    """Median of the *measured* (non-zero) values of each column where 0 means missing.

    Computed on the training split only, so no information leaks from the
    validation or test patients into the model.
    """
    return {
        col: float(X_train.loc[X_train[col] != 0, col].median())
        for col in config.ZERO_MEANS_MISSING
    }


def apply_imputer(X: pd.DataFrame, medians: dict) -> pd.DataFrame:
    X = X.astype(float).copy()
    for col, median in medians.items():
        X.loc[X[col] == 0, col] = median
    return X


def load_dataset(seed: int = config.SEED, path=config.DATA_PATH) -> Dataset:
    """Stratified 60/20/20 split, then median imputation fitted on the training split."""
    df = load_raw(path)
    X = df[config.FEATURES].astype(float)
    y = df[config.TARGET].to_numpy(dtype=int)

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed
    )
    val_fraction = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, stratify=y_rest, random_state=seed
    )

    medians = fit_imputer(X_train)
    return Dataset(
        X_train=apply_imputer(X_train, medians).to_numpy(),
        y_train=y_train,
        X_val=apply_imputer(X_val, medians).to_numpy(),
        y_val=y_val,
        X_test=apply_imputer(X_test, medians).to_numpy(),
        y_test=y_test,
        feature_names=list(config.FEATURES),
        medians=medians,
    )
