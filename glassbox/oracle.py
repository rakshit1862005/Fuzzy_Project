"""Querying the network as an oracle to get more training data for the surrogate.

The fuzzy system's job is to copy the *network*, not the real labels, so we
are not limited to the 460 real training patients: we can invent nearby
patients and simply ask the network what it thinks of them (the idea behind
TREPAN, Craven & Shavlik 1996). More examples near the decision boundary make
the copy more faithful.
"""
import numpy as np

from . import config


def oracle_augment(X, oracle_fn, n_samples=config.ORACLE_SAMPLES, noise=config.ORACLE_NOISE,
                   seed=config.SEED):
    """Return (synthetic inputs, network probabilities for them).

    Each synthetic point is a random real training patient plus Gaussian
    noise (std = `noise` x the feature's std), clipped to the observed range
    so we never ask about impossible patients.
    """
    X = np.asarray(X, dtype=float)
    if n_samples <= 0:
        return np.empty((0, X.shape[1])), np.empty(0)
    rng = np.random.default_rng(seed)
    base = X[rng.integers(0, len(X), n_samples)]
    synthetic = base + rng.normal(0.0, 1.0, base.shape) * noise * X.std(axis=0)
    synthetic = np.clip(synthetic, X.min(axis=0), X.max(axis=0))
    return synthetic, oracle_fn(synthetic)
