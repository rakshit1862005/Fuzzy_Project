"""Gaussian membership functions and linguistic term names."""
import numpy as np

TERM_NAMES = {
    1: ["Any"],
    2: ["Low", "High"],
    3: ["Low", "Medium", "High"],
    4: ["Very Low", "Low", "High", "Very High"],
    5: ["Very Low", "Low", "Medium", "High", "Very High"],
}

# A Gaussian falls to 0.5 at a distance of sqrt(2 ln 2) * sigma from its centre.
HALF_POINT = np.sqrt(2.0 * np.log(2.0))


def gaussian(x, center, sigma):
    """mu(x) = exp(-(x - c)^2 / (2 sigma^2)), a smooth bell with peak 1 at x = c."""
    return np.exp(-0.5 * ((np.asarray(x) - center) / sigma) ** 2)


def term_offsets(X, centers):
    """Distance of each value from each term centre, with "shoulders" at both ends.

    X: (n, k) inputs, centers: (k, m) -> (n, k, m). The first term (Low) is
    flat at membership 1 below its centre and the last term (High) is flat at
    1 above it, so a glucose of 200 is fully "High" rather than "not quite High".
    Inner terms (Medium) are ordinary bells.
    """
    d = np.asarray(X, dtype=float)[:, :, None] - np.asarray(centers)[None]
    if d.shape[2] > 1:
        d[:, :, 0] = np.maximum(d[:, :, 0], 0.0)      # below the Low centre: offset 0, mu = 1
        d[:, :, -1] = np.minimum(d[:, :, -1], 0.0)    # above the High centre: offset 0, mu = 1
    return d


def log_memberships(X, centers, sigmas):
    """log mu for every input, value and term: -(offset)^2 / (2 sigma^2)."""
    return -0.5 * (term_offsets(X, centers) / np.asarray(sigmas)[None]) ** 2


def term_names(n_terms: int) -> list:
    if n_terms not in TERM_NAMES:
        raise ValueError(f"n_terms must be one of {sorted(TERM_NAMES)}")
    return TERM_NAMES[n_terms]


def initial_partition(column, n_terms: int):
    """Initial (centres, sigmas) for one input already scaled to [0, 1].

    Centres sit at evenly spaced quantiles of the data (10 % ... 90 %), so each
    term covers a similar number of patients even for skewed features like
    Insulin. Sigmas are chosen so that neighbouring terms cross at membership
    0.5 - every value belongs at least "half" to some term.
    """
    column = np.asarray(column, dtype=float)
    if n_terms == 1:
        return np.array([0.5]), np.array([0.5])
    centers = np.quantile(column, np.linspace(0.1, 0.9, n_terms))
    if np.min(np.diff(centers)) < 0.05:          # heavily tied data: fall back to even spacing
        centers = np.linspace(0.1, 0.9, n_terms)
    gap = np.mean(np.diff(centers))
    sigmas = np.full(n_terms, gap / (2.0 * HALF_POINT))
    return centers, sigmas
