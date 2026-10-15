"""Which inputs does the network actually rely on? (Permutation importance.)

For each feature we shuffle its column - destroying its information while
keeping its distribution - and measure how much the network's output moves.
A feature the network ignores barely changes the output when shuffled.
This only needs the network's predictions, never its weights.
"""
import numpy as np
import pandas as pd

from . import config


def permutation_importance(predict_proba, X, feature_names, n_repeats=30, seed=config.SEED) -> pd.DataFrame:
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    base = predict_proba(X)
    rows = []
    for j, name in enumerate(feature_names):
        changes = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            changes.append(np.mean(np.abs(predict_proba(Xp) - base)))
        rows.append({"feature": name, "importance": float(np.mean(changes)), "std": float(np.std(changes))})
    table = pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)
    table["share"] = table["importance"] / table["importance"].sum()
    return table


def top_features(importance: pd.DataFrame, feature_names, k: int) -> list:
    """Indices (into feature_names) of the k most important features, most important first."""
    return [list(feature_names).index(f) for f in importance["feature"].head(k)]
