"""Explaining one prediction: which rules fired, how strongly, and why.

Because the fuzzy output is y = sum_r wn_r * p_r, each rule's share of the
final answer is exactly wn_r * p_r. These contributions add up to y, so the
explanation is complete - nothing is hidden.

Normalisation has one catch: the weights always sum to 1, even for an
unusual patient whom no rule describes well. So we also report the *match
strength* - the raw firing strength of the best rule. A low value means "the
rule base has no good description of this patient", and the explanation
says so instead of sounding confident.
"""
import numpy as np

from . import config
from .rules import condition_text, risk_label, rule_text
from .rulebase import DONT_CARE

WEAK_MATCH = 0.2    # best rule fires below this -> explanation flagged as weak


def explain_instance(model, x, nn_prob=None, top_k=None) -> dict:
    x = np.asarray(x, dtype=float).reshape(1, -1)
    mu = model.memberships(x)[0]                    # (n_features, n_terms)
    w, wbar = model.firing_strengths(x)
    w, wbar = w[0], wbar[0]
    f = model.rule_outputs(x)[0]                    # what each rule says for this patient
    output = float(model.predict_proba(x)[0])

    rules = []
    for r in np.argsort(-wbar):
        A = model.antecedents_[r]
        rules.append({
            "rule": int(r + 1),
            "text": rule_text(model, r),
            "conditions": [
                {"condition": condition_text(model, j, A[j], with_value=False),
                 "membership": float(mu[j, A[j]])}
                for j in range(len(A)) if A[j] != DONT_CARE
            ],
            "firing": float(w[r]),
            "weight": float(wbar[r]),
            "consequent": float(f[r]),
            "contribution": float(wbar[r] * f[r]),
        })
    if top_k is not None:
        rules = rules[:top_k]

    memberships = {
        name: {term: float(mu[j, t]) for t, term in enumerate(model.term_names)}
        for j, name in enumerate(model.feature_names)
    }
    out = {
        "inputs": dict(zip(model.feature_names, x[0].tolist())),
        "memberships": memberships,
        "rules": rules,
        "fuzzy_output": output,
        "fuzzy_decision": config.CLASS_NAMES[int(output >= model.threshold)],
        "match_strength": float(w.max()),
        "weak_match": bool(w.max() < WEAK_MATCH),
    }
    if nn_prob is not None:
        out["nn_prob"] = float(nn_prob)
        out["nn_decision"] = config.CLASS_NAMES[int(nn_prob >= model.threshold)]
        out["agree"] = out["nn_decision"] == out["fuzzy_decision"]
    return out


def explanation_markdown(expl, title="Case", top_k=3, true_label=None) -> str:
    lines = [f"### {title}", ""]
    inputs = ", ".join(
        f"{k} = {v:.0f}" if abs(v) >= 10 else f"{k} = {v:.2f}" for k, v in expl["inputs"].items()
    )
    lines.append(f"**Inputs used by the fuzzy system:** {inputs}")
    lines.append("")
    if "nn_prob" in expl:
        lines.append(f"- Neural network: P(diabetic) = {expl['nn_prob']:.2f} -> **{expl['nn_decision']}**")
    lines.append(f"- Fuzzy system: risk = {expl['fuzzy_output']:.2f} -> **{expl['fuzzy_decision']}**"
                 + ("" if "agree" not in expl else (" (agrees)" if expl["agree"] else " (DISAGREES)")))
    if true_label is not None:
        lines.append(f"- Actual outcome: {config.CLASS_NAMES[int(true_label)]}")
    lines.append(f"- Match strength (best rule's raw firing): {expl['match_strength']:.2f}"
                 + (" - **weak: no rule describes this patient well**" if expl["weak_match"] else ""))
    lines += ["", "**Fuzzified inputs (degree of membership):**", ""]
    for name, terms in expl["memberships"].items():
        lines.append("- " + name + ": " + ", ".join(f"{t} {m:.2f}" for t, m in terms.items()))
    lines += ["", f"**Top {top_k} rules by weight:**", "",
              "| Rule | Conditions (membership) | Weight | Rule output | Contribution |",
              "|---|---|---|---|---|"]
    for rule in expl["rules"][:top_k]:
        conds = "; ".join(f"{c['condition']} ({c['membership']:.2f})" for c in rule["conditions"]) or "(always)"
        lines.append(
            f"| {rule['rule']} | {conds} | {rule['weight']:.2f} | "
            f"{rule['consequent']:.2f} ({risk_label(rule['consequent'])}) | {rule['contribution']:.3f} |"
        )
    rest = sum(r["contribution"] for r in expl["rules"][top_k:])
    lines += ["", f"Other rules contribute {rest:.3f}; all contributions sum to the fuzzy output "
                  f"{expl['fuzzy_output']:.2f}."]
    return "\n".join(lines)
