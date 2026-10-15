"""Fuzzy C-Means clustering (Bezdek, 1981), written from scratch with NumPy.

Unlike k-means, every point belongs to every cluster with a degree u in [0, 1]
and the degrees of one point sum to 1. The algorithm alternates two steps:

    centres:      v_i  = sum_k u_ik^m x_k / sum_k u_ik^m
    memberships:  u_ik = 1 / sum_j (d_ik / d_jk)^(2 / (m - 1))

where d_ik is the distance from point k to centre i and m > 1 is the
"fuzzifier" (m = 2 is the usual choice; larger m gives softer clusters).
"""
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class FCMResult:
    centers: np.ndarray      # (c, d)
    U: np.ndarray            # (n, c) membership of each point in each cluster
    n_iter: int
    objective: list          # J_m after every iteration (should decrease)


def _memberships(X, centers, m):
    dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    dist = np.fmax(dist, 1e-12)                        # a point exactly on a centre
    ratio = (dist[:, :, None] / dist[:, None, :]) ** (2.0 / (m - 1.0))
    return 1.0 / ratio.sum(axis=2), dist


def fuzzy_c_means(X, n_clusters, m=2.0, max_iter=300, tol=1e-6, seed=config.SEED) -> FCMResult:
    X = np.asarray(X, dtype=float)
    if not 1 <= n_clusters <= len(X):
        raise ValueError("n_clusters must be between 1 and the number of points")
    rng = np.random.default_rng(seed)
    U = rng.dirichlet(np.ones(n_clusters), size=len(X))   # random rows that sum to 1
    objective = []
    for it in range(1, max_iter + 1):
        Um = U ** m
        centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
        U_new, dist = _memberships(X, centers, m)
        objective.append(float(np.sum((U_new ** m) * dist ** 2)))
        converged = np.max(np.abs(U_new - U)) < tol
        U = U_new
        if converged:
            break
    return FCMResult(centers=centers, U=U, n_iter=it, objective=objective)


def partition_coefficient(U) -> float:
    """Bezdek's fuzzy partition coefficient: 1 = crisp clusters, 1/c = totally fuzzy."""
    return float(np.mean(np.sum(U ** 2, axis=1)))
