"""Building the IF-part (antecedents) of a fuzzy rule base.

A rule base is stored as an integer matrix A of shape (n_rules, n_features):
A[r, j] = t means "feature j is term t" in rule r, and A[r, j] = DONT_CARE (-1)
means rule r does not mention feature j at all.

Example with terms (Low, Medium, High) = (0, 1, 2):
    [2, -1, 1]  ->  IF x0 is High AND x2 is Medium THEN ...
"""
import itertools

import numpy as np

from . import config
from .fcm import fuzzy_c_means
from .membership import gaussian

DONT_CARE = -1


def grid_rules(n_features: int, n_terms: int) -> np.ndarray:
    """Every combination of terms: n_terms ** n_features rules (the classic ANFIS grid).

    This explodes quickly - 3 terms on 8 features is already 6561 rules -
    which is why we use clustering instead.
    """
    return np.array(list(itertools.product(range(n_terms), repeat=n_features)), dtype=int)


def fcm_rules(X_norm, target, centers, sigmas, n_clusters, seed=config.SEED) -> np.ndarray:
    """One candidate rule per Fuzzy C-Means cluster, snapped to the shared linguistic terms.

    We cluster the inputs *together with the network's output*, so clusters
    group patients that look alike AND receive a similar decision. Each
    cluster centre is then translated into words: for every feature we pick
    the term (Low/Medium/High) with the highest membership at the centre.
    Clusters that translate to the same sentence are merged, so the result
    can have fewer than `n_clusters` rules.
    """
    X_norm = np.asarray(X_norm, dtype=float)
    data = np.column_stack([X_norm, np.asarray(target, dtype=float)])
    result = fuzzy_c_means(data, n_clusters, seed=seed)

    # Biggest clusters first, so merging keeps the most representative ones.
    order = np.argsort(-result.U.sum(axis=0))
    rules = []
    for i in order:
        center = result.centers[i, :-1]
        mu = gaussian(center[:, None], centers, sigmas)       # (n_features, n_terms)
        rule = tuple(int(t) for t in mu.argmax(axis=1))
        if rule not in rules:
            rules.append(rule)
    return np.array(rules, dtype=int)


def unique_rules(A: np.ndarray) -> np.ndarray:
    """Drop duplicate rows while keeping the original order."""
    _, first = np.unique(A, axis=0, return_index=True)
    return A[np.sort(first)]
