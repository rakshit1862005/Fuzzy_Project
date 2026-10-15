"""Interactive demo of the Glass-Box Insight project.

    streamlit run app.py

Pick a test patient (or type in your own values) and see the neural network's
prediction next to the fuzzy rules that explain it. For teaching only - this
is not a medical tool.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from glassbox import config, plots
from glassbox.explain import explain_instance
from glassbox.metrics import anfis_complexity, fidelity
from glassbox.pipeline import train_glassbox
from glassbox.rules import partition_table, risk_label, rule_table

st.set_page_config(page_title="Glass-Box Insight", layout="wide")


@st.cache_resource(show_spinner="Training the neural network and the fuzzy explainer (about a minute) ...")
def build():
    params_file = config.RESULTS_DIR / "nn_params.json"
    params = json.loads(params_file.read_text()) if params_file.exists() else None
    return train_glassbox(nn_params=params)


gb = build()
data, bb, sur = gb.data, gb.blackbox, gb.surrogate
model, selected = sur.model, sur.selected

st.title("Glass-Box Insight")
st.caption("A neural network predicts diabetes; a fuzzy IF-THEN rule base explains how it decides. "
           "Educational demo on the Pima Indians Diabetes dataset - not medical advice.")

# ------------------------------------------------------------------ input
with st.sidebar:
    st.header("Patient")
    mode = st.radio("Input", ["Test-set patient", "Enter values"], horizontal=True)
    if mode == "Test-set patient":
        i = st.number_input(f"Patient index (0 - {len(data.y_test) - 1})", 0, len(data.y_test) - 1, 0)
        x = data.X_test[i].copy()
        st.write(f"Actual outcome: **{config.CLASS_NAMES[data.y_test[i]]}**")
        st.dataframe(pd.DataFrame({"value": x}, index=data.feature_names))
    else:
        x = np.empty(len(data.feature_names))
        lo, hi = data.X_train.min(axis=0), data.X_train.max(axis=0)
        median = np.median(data.X_train, axis=0)
        for j, name in enumerate(data.feature_names):
            unit = config.UNITS[name]
            label = f"{name}{' (' + unit + ')' if unit else ''}" + ("  - used by fuzzy rules" if j in selected else "")
            step = 0.01 if name == "DiabetesPedigreeFunction" else 1.0
            x[j] = st.slider(label, float(lo[j]), float(hi[j]), float(round(median[j], 2)), step=step)

# ---------------------------------------------------------------- outputs
nn_prob = float(bb.predict_proba(x[None])[0])
expl = explain_instance(model, x[selected], nn_prob=nn_prob)

c1, c2, c3 = st.columns(3)
c1.metric("Neural network: P(diabetic)", f"{nn_prob:.2f}", expl["nn_decision"], delta_color="off")
c2.metric("Fuzzy rules: risk", f"{expl['fuzzy_output']:.2f}", expl["fuzzy_decision"], delta_color="off")
c3.metric("Do they agree?", "Yes" if expl["agree"] else "No",
          f"test-set fidelity {fidelity(model.predict(data.X_test[:, selected]), bb.predict(data.X_test)):.1%}",
          delta_color="off")

if expl["weak_match"]:
    st.warning(f"Weak match: the best rule fits this patient only to degree {expl['match_strength']:.2f}. "
               "The rule base has no good description of such a patient, so treat the explanation with care.")

why, rules_tab, terms_tab, about = st.tabs(["Why this decision?", "All rules", "Linguistic terms", "About"])

with why:
    st.pyplot(plots.case_contributions(expl, None))
    st.markdown("**Rules that fired** - weight = share of the decision; contribution = weight x rule risk. "
                "Contributions add up exactly to the fuzzy risk.")
    st.dataframe(pd.DataFrame([{
        "rule": f"R{r['rule']}",
        "IF": " AND ".join(f"{c['condition']} ({c['membership']:.2f})" for c in r["conditions"]) or "(always)",
        "THEN risk": round(r["consequent"], 2),
        "meaning": risk_label(r["consequent"]),
        "weight": round(r["weight"], 3),
        "contribution": round(r["contribution"], 3),
    } for r in expl["rules"]]), hide_index=True)
    st.markdown("**Fuzzification** - how strongly each value belongs to each term:")
    st.dataframe(pd.DataFrame(expl["memberships"]).T.round(2))

with rules_tab:
    cx = anfis_complexity(model)
    st.write(f"{cx['rules']} rules, {cx['conditions']} conditions in total "
             f"(before simplification: {anfis_complexity(sur.full)['conditions']}). "
             f"Inputs: {', '.join(gb.selected_names)}.")
    st.dataframe(rule_table(model, data.X_train[:, selected]), hide_index=True)
    with st.expander("Simplification log"):
        st.text("\n".join(sur.simplify_log))

with terms_tab:
    st.pyplot(plots.membership_functions(model, None, x=x[selected]))
    st.dataframe(partition_table(model), hide_index=True)

with about:
    st.markdown(f"""
**Black box:** {bb.describe()}.

**Explainer:** zero-order Takagi-Sugeno ANFIS on the network's {len(selected)} most important inputs,
rules initialised by Fuzzy C-Means, trained by hybrid learning (least squares + gradient descent)
to copy the network's probabilities, then pruned.

Full results, plots and the trade-off study: `results/summary.md` (regenerate with
`python scripts/run_experiments.py`).
""")
    tradeoff_png = config.RESULTS_DIR / "figures" / "tradeoff.png"
    if tradeoff_png.exists():
        st.image(str(tradeoff_png))
