import numpy as np

from glassbox.fcm import fuzzy_c_means, partition_coefficient


def blobs(seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(0, 0.1, (50, 2)), rng.normal(3, 0.1, (50, 2))])


def test_memberships_sum_to_one():
    result = fuzzy_c_means(blobs(), 3, seed=1)
    assert np.allclose(result.U.sum(axis=1), 1.0)
    assert np.all((result.U >= 0) & (result.U <= 1))


def test_finds_two_separated_blobs():
    result = fuzzy_c_means(blobs(), 2, seed=1)
    centers = result.centers[np.argsort(result.centers[:, 0])]
    assert np.allclose(centers, [[0, 0], [3, 3]], atol=0.1)
    assert partition_coefficient(result.U) > 0.9    # nearly crisp


def test_objective_does_not_increase():
    obj = np.array(fuzzy_c_means(blobs(), 2, seed=3).objective)
    assert np.all(np.diff(obj) <= 1e-9)
