import numpy as np
import pytest

from glassbox.anfis import ANFIS
from glassbox.membership import gaussian, initial_partition
from glassbox.rulebase import DONT_CARE, grid_rules


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, (n, 2))
    y = (X[:, 0] > 6).astype(float)                # only x0 matters
    return X, y


def test_initial_terms_cross_at_half_membership():
    centers, sigmas = initial_partition(np.linspace(0, 1, 101), 3)
    midpoint = (centers[0] + centers[1]) / 2
    assert gaussian(midpoint, centers[0], sigmas[0]) == pytest.approx(0.5)


def test_grid_rule_count():
    assert grid_rules(3, 2).shape == (8, 3)
    assert grid_rules(4, 3).shape == (81, 4)


def test_normalised_weights_sum_to_one():
    X, y = toy()
    model = ANFIS(["a", "b"], n_terms=2, rule_init="grid", epochs=5).fit(X, y)
    _, wbar = model.firing_strengths(X)
    assert np.allclose(wbar.sum(axis=1), 1.0)


def test_output_is_weighted_average_of_rule_outputs():
    X, y = toy()
    model = ANFIS(["a", "b"], n_terms=2, rule_init="grid", epochs=5).fit(X, y)
    _, wbar = model.firing_strengths(X[:5])
    assert np.allclose(wbar @ model.consequents_, model.predict_proba(X[:5]))


def test_analytic_gradient_matches_numerical():
    X, y = toy(60)
    model = ANFIS(["a", "b"], n_terms=3, rule_init="grid", epochs=3).fit(X, y)
    Xn = model._scale(X)

    def loss(centers, sigmas):
        wbar = model._normalize(model._log_firing(Xn, centers, sigmas))
        return np.mean((wbar @ model.consequents_ - y) ** 2)

    wbar = model._normalize(model._log_firing(Xn))
    grad_c, grad_s = model._premise_gradients(Xn, wbar, wbar @ model.consequents_, y)
    eps = 1e-6
    for j in range(2):
        for t in range(3):
            for param, grad in ((model.centers_, grad_c), (model.sigmas_, grad_s)):
                plus, minus = param.copy(), param.copy()
                plus[j, t] += eps
                minus[j, t] -= eps
                if param is model.centers_:
                    numeric = (loss(plus, model.sigmas_) - loss(minus, model.sigmas_)) / (2 * eps)
                else:
                    numeric = (loss(model.centers_, plus) - loss(model.centers_, minus)) / (2 * eps)
                assert grad[j, t] == pytest.approx(numeric, rel=1e-4, abs=1e-8)


def test_learns_a_threshold_rule():
    X, y = toy()
    model = ANFIS(["a", "b"], n_terms=2, rule_init="grid", epochs=150).fit(X, y)
    Xt, yt = toy(seed=1)
    assert np.mean(model.predict(Xt) == yt) > 0.9


def test_constraints_keep_terms_ordered_and_close_to_start():
    X, y = toy()
    model = ANFIS(["a", "b"], n_terms=3, rule_init="grid", epochs=200, lr=0.1).fit(X, y)
    assert np.all(np.diff(model.centers_, axis=1) > 0)
    assert np.all(np.abs(model.centers_ - model.initial_centers_) <= model.max_shift + model.min_gap + 1e-9)
    assert np.all(model.sigmas_ >= model.initial_sigmas_ * model.sigma_range[0] - 1e-12)
    assert np.all(model.sigmas_ <= model.initial_sigmas_ * model.sigma_range[1] + 1e-12)


def test_dont_care_condition_ignores_feature():
    X, y = toy()
    rules = np.array([[0, DONT_CARE], [1, DONT_CARE]])
    model = ANFIS(["a", "b"], n_terms=2, antecedents=rules, epochs=50).fit(X, y)
    a, b = X[:1].copy(), X[:1].copy()
    b[0, 1] += 5.0
    assert model.predict_proba(a) == pytest.approx(model.predict_proba(b))


def test_fcm_rule_init_gives_at_most_n_rules():
    X, y = toy()
    model = ANFIS(["a", "b"], n_terms=3, n_rules=5, epochs=5).fit(X, y)
    assert 1 <= len(model.antecedents_) <= 5
    assert len(np.unique(model.antecedents_, axis=0)) == len(model.antecedents_)
