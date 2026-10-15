import numpy as np

from glassbox.anfis import ANFIS
from glassbox.explain import explain_instance, explanation_markdown
from glassbox.metrics import anfis_complexity, fidelity, fidelity_by_confidence
from glassbox.rules import rule_table, rule_text
from glassbox.simplify import simplify


def model_and_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 200, (300, 2))
    y = (X[:, 0] > 120).astype(float)
    model = ANFIS(["Glucose", "BMI"], n_terms=3, rule_init="grid", epochs=60).fit(X, y)
    return model, X, y


def test_fidelity():
    assert fidelity([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75
    split = fidelity_by_confidence([1, 0, 1], [1, 1, 1], [0.9, 0.55, 0.1])
    assert split["n_boundary"] == 1 and split["boundary"] == 0.0 and split["confident"] == 1.0


def test_rule_text_reads_like_a_sentence():
    model, X, _ = model_and_data()
    text = rule_text(model, 0)
    assert text.startswith("IF Glucose is Low") and " AND BMI is Low" in text and "THEN risk =" in text
    assert len(rule_table(model, X)) == len(model.antecedents_)


def test_explanation_contributions_sum_to_output():
    model, X, _ = model_and_data()
    expl = explain_instance(model, X[0], nn_prob=0.7)
    total = sum(r["contribution"] for r in expl["rules"])
    assert np.isclose(total, model.predict_proba(X[:1])[0], atol=1e-9)
    assert "Top 3 rules" in explanation_markdown(expl)


def test_simplify_reduces_complexity_without_losing_fidelity():
    model, X, y = model_and_data()
    labels = y.astype(int)
    simple, log = simplify(model, X, y, X, labels, tolerance=0.01)
    before, after = anfis_complexity(model), anfis_complexity(simple)
    assert after["conditions"] < before["conditions"]
    assert fidelity(simple.predict(X), labels) >= fidelity(model.predict(X), labels) - 0.01
    assert log[0].startswith("start")
