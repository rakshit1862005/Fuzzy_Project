"""Turning a trained ANFIS into readable sentences and tables."""
import numpy as np
import pandas as pd

from . import config
from .rulebase import DONT_CARE


def _fmt(value):
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.2f}"


def risk_label(p) -> str:
    """Words for a rule output; the decision itself still uses the 0.5 threshold."""
    if p < 0.2:
        return "very low risk"
    if p < 0.4:
        return "low risk"
    if p < 0.6:
        return "uncertain"
    if p < 0.8:
        return "high risk"
    return "very high risk"


def condition_text(model, j, t, with_value=True) -> str:
    name = model.feature_names[j]
    text = f"{name} is {model.term_names[t]}"
    if with_value:
        unit = config.UNITS.get(name, "")
        text += f" (~{_fmt(model.centers_original[j, t])}{(' ' + unit) if unit else ''})"
    return text


def linear_then_part(model, r) -> str:
    """A first-order THEN-part written in original units: b + a1*x1 + ...

    The model stores coefficients for inputs scaled to [0, 1]; we convert them
    back so the formula uses mg/dL, years, etc.
    """
    coef = model.consequents_[r, :-1] / model.span_
    intercept = model.consequents_[r, -1] - np.sum(coef * model.lo_)
    terms = [f"{intercept:.3g}"]
    for a, name in zip(coef, model.feature_names):
        terms.append(f"{'-' if a < 0 else '+'} {abs(a):.3g}*{name}")
    return " ".join(terms)


def rule_text(model, r, with_value=True) -> str:
    A = model.antecedents_
    conditions = [condition_text(model, j, A[r, j], with_value)
                  for j in range(A.shape[1]) if A[r, j] != DONT_CARE]
    if_part = " AND ".join(conditions) if conditions else "(always)"
    if model.order == 1:
        return f"IF {if_part} THEN risk = {linear_then_part(model, r)}"
    p = model.consequents_[r]
    decision = config.CLASS_NAMES[int(p >= model.threshold)]
    return f"IF {if_part} THEN risk = {p:.2f} ({risk_label(p)} -> {decision})"


def rule_table(model, X) -> pd.DataFrame:
    """One row per rule with its coverage on data X.

    coverage  = share of patients for whom this rule has the largest weight
    avg_weight = mean normalised firing strength (its average influence)
    """
    _, wbar = model.firing_strengths(X)
    winner = wbar.argmax(axis=1)
    rows = []
    for r in range(len(model.antecedents_)):
        if model.order == 1:
            then, meaning, decision = linear_then_part(model, r), "depends on inputs", "depends on inputs"
        else:
            p = float(model.consequents_[r])
            then, meaning, decision = round(p, 3), risk_label(p), config.CLASS_NAMES[int(p >= model.threshold)]
        rows.append({
            "rule": r + 1,
            "IF": rule_text(model, r).split(" THEN ")[0][3:],
            "THEN risk": then,
            "meaning": meaning,
            "decision": decision,
            "coverage": round(float(np.mean(winner == r)), 3),
            "avg_weight": round(float(wbar[:, r].mean()), 3),
        })
    return pd.DataFrame(rows)


def partition_table(model) -> pd.DataFrame:
    """Every linguistic term with its centre and width in original units."""
    rows = []
    for j, name in enumerate(model.feature_names):
        for t, term in enumerate(model.term_names):
            rows.append({
                "feature": name,
                "term": term,
                "center": round(float(model.centers_original[j, t]), 2),
                "sigma": round(float(model.sigmas_original[j, t]), 2),
                "unit": config.UNITS.get(name, ""),
            })
    return pd.DataFrame(rows)


def rules_markdown(model, X) -> str:
    lines = []
    table = rule_table(model, X)
    for r, row in table.iterrows():
        lines.append(f"**Rule {r + 1}** (covers {row['coverage']:.0%} of patients): {rule_text(model, r)}")
    return "\n\n".join(lines)
