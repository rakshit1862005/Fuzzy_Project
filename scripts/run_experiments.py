"""Run every experiment in the project and write tables, figures and a summary to results/.

    python scripts/run_experiments.py            # full run (~10 minutes on a laptop)
    python scripts/run_experiments.py --quick    # 1 seed instead of 5 (~2 minutes)
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glassbox import config, plots  # noqa: E402
from glassbox.blackbox import tune_mlp  # noqa: E402
from glassbox.data import load_dataset  # noqa: E402
from glassbox.explain import explain_instance, explanation_markdown  # noqa: E402
from glassbox.feature_selection import permutation_importance, top_features  # noqa: E402
from glassbox.metrics import (  # noqa: E402
    anfis_complexity,
    classification_metrics,
    fidelity,
    fidelity_by_confidence,
    probability_mae,
)
from glassbox.pipeline import BlackBox, fit_fuzzy_surrogate  # noqa: E402
from glassbox.rules import partition_table, rule_table, rules_markdown  # noqa: E402
from glassbox.surrogates import fit_tree_surrogate, tree_complexity, tree_rules_text  # noqa: E402

OUT = config.RESULTS_DIR
FIG = OUT / "figures"


def md_table(df: pd.DataFrame, floatfmt=".3f") -> str:
    """DataFrame -> GitHub markdown table without extra dependencies."""
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def evaluate_surrogate(sur, data, bb):
    Xt = data.X_test[:, sur.selected]
    nn_prob, nn_lab = bb.predict_proba(data.X_test), bb.predict(data.X_test)
    out = {}
    for stage, model in (("full", sur.full), ("simplified", sur.model)):
        pred = model.predict(Xt)
        split = fidelity_by_confidence(pred, nn_lab, nn_prob)
        out[stage] = {
            **anfis_complexity(model),
            "fidelity": fidelity(pred, nn_lab),
            "fidelity_boundary": split["boundary"],
            "fidelity_confident": split["confident"],
            "prob_mae": probability_mae(model.predict_proba(Xt), nn_prob),
            "accuracy": classification_metrics(data.y_test, model.predict_proba(Xt))["accuracy"],
        }
    return out


def main(quick=False):
    warnings.filterwarnings("ignore")
    FIG.mkdir(parents=True, exist_ok=True)
    seeds = config.SWEEP_SEEDS[:1] if quick else config.SWEEP_SEEDS
    t0 = time.time()
    report = {}

    # ------------------------------------------------ 1. black-box network
    print("1/7 tuning the neural network ...")
    data = load_dataset()
    pipeline, nn_params, cv = tune_mlp(data.X_train, data.y_train)
    bb = BlackBox(pipeline)
    nn_test = classification_metrics(data.y_test, bb.predict_proba(data.X_test))
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(data.X_train, data.y_train)
    lr_test = classification_metrics(data.y_test, logreg.predict_proba(data.X_test)[:, 1])
    cv_table = pd.DataFrame({
        "hidden layers": [str(p["mlp__hidden_layer_sizes"]) for p in cv["params"]],
        "L2 alpha": [p["mlp__alpha"] for p in cv["params"]],
        "CV ROC-AUC": cv["mean_test_score"],
        "std": cv["std_test_score"],
    }).sort_values("CV ROC-AUC", ascending=False)
    report["blackbox"] = {"params": {"hidden_layer_sizes": list(nn_params["hidden_layer_sizes"]),
                                     "alpha": nn_params["alpha"]},
                          "description": bb.describe(), "test": nn_test, "logreg_test": lr_test}
    (OUT / "nn_params.json").write_text(json.dumps(report["blackbox"]["params"], indent=2))

    # --------------------------------------------- 2. feature importance
    print("2/7 permutation importance ...")
    importance = permutation_importance(bb.predict_proba, data.X_val, data.feature_names)
    selected = top_features(importance, data.feature_names, config.N_FEATURES)
    names = [data.feature_names[j] for j in selected]
    plots.feature_importance(importance, FIG / "feature_importance.png", names)

    # ------------------------------------------------ 3. main fuzzy model
    print("3/7 training the main fuzzy explainer ...")
    sur = fit_fuzzy_surrogate(data, bb, selected)
    report["main"] = evaluate_surrogate(sur, data, bb)
    model = sur.model
    Xtr = data.X_train[:, selected]
    plots.membership_functions(model, FIG / "membership_functions.png")
    plots.training_curve(sur.full.history_, FIG / "training_curve.png")
    rules_doc = [
        "# Extracted fuzzy rule base", "",
        f"Inputs: {', '.join(names)} - {config.N_TERMS} terms each "
        f"({', '.join(model.term_names)}).", "",
        "## Final (simplified) rules", "", rules_markdown(model, Xtr), "",
        md_table(rule_table(model, Xtr)), "",
        "## Linguistic terms (original units)", "", md_table(partition_table(model), ".2f"), "",
        "## Rules before simplification", "", rules_markdown(sur.full, Xtr), "",
        "## Simplification log", "", *[f"- {line}" for line in sur.simplify_log],
    ]
    (OUT / "rules.md").write_text("\n".join(rules_doc) + "\n")

    # --------------------------------------------------- 4. case studies
    print("4/7 case studies ...")
    Xt = data.X_test[:, selected]
    nn_prob = bb.predict_proba(data.X_test)
    fz_pred = model.predict(Xt)
    agree = fz_pred == bb.predict(data.X_test)
    cases = {
        "confident_diabetic": int(np.argmax(np.where(agree, nn_prob, -1))),
        "confident_healthy": int(np.argmin(np.where(agree, nn_prob, 2))),
        "borderline": int(np.argmin(np.abs(nn_prob - 0.5))),
    }
    if (~agree).any():
        cases["disagreement"] = int(np.flatnonzero(~agree)[np.argmax(np.abs(nn_prob[~agree] - 0.5))])
    case_doc = ["# Case studies (test-set patients)", "",
                "Each case shows the patient, both models' outputs, how each input was fuzzified, "
                "and which rules produced the answer.", ""]
    for name, i in cases.items():
        expl = explain_instance(model, Xt[i], nn_prob=nn_prob[i])
        title = f"{name.replace('_', ' ').title()} (test patient #{i})"
        case_doc += [explanation_markdown(expl, title, true_label=data.y_test[i]), "",
                     f"![{name}](figures/case_{name}.png)", ""]
        plots.case_contributions(expl, FIG / f"case_{name}.png")
    (OUT / "case_studies.md").write_text("\n".join(case_doc))

    # ----------------------------------------------- 5. trade-off sweep
    print(f"5/7 trade-off sweep ({len(config.SWEEP_RULES)} sizes x {len(seeds)} seeds) ...")
    rows = []
    nn_lab = bb.predict(data.X_test)
    for n_rules in config.SWEEP_RULES:
        for seed in seeds:
            s = fit_fuzzy_surrogate(data, bb, selected, n_rules=n_rules, seed=seed)
            ev = evaluate_surrogate(s, data, bb)
            for stage, label in (("full", "ANFIS (FCM rules)"), ("simplified", "ANFIS simplified")):
                rows.append({"method": label, "target_rules": n_rules, "seed": seed, **ev[stage]})
        tree = fit_tree_surrogate(data.X_train[:, selected], bb.predict(data.X_train), n_rules)
        rows.append({"method": "Decision tree", "target_rules": n_rules, "seed": 0,
                     **tree_complexity(tree),
                     "fidelity": fidelity(tree.predict(Xt), nn_lab),
                     "accuracy": float(np.mean(tree.predict(Xt) == data.y_test))})
    for n_terms in (2, 3):
        s = fit_fuzzy_surrogate(data, bb, selected, n_terms=n_terms, rule_init="grid", do_simplify=False)
        rows.append({"method": f"ANFIS grid ({n_terms} terms)", "target_rules": n_terms ** len(selected),
                     "seed": 0, **evaluate_surrogate(s, data, bb)["full"]})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "tradeoff.csv", index=False)
    curve = (sweep[sweep["method"].isin(["ANFIS (FCM rules)", "ANFIS simplified", "Decision tree"])]
             .groupby(["method", "target_rules"], sort=False)
             .agg(rules_mean=("rules", "mean"), conditions_mean=("conditions", "mean"),
                  fidelity_mean=("fidelity", "mean"), fidelity_std=("fidelity", "std"),
                  accuracy_mean=("accuracy", "mean"))
             .reset_index().fillna({"fidelity_std": 0.0}))
    plots.tradeoff(curve, FIG / "tradeoff.png")

    # ------------------------------------------------------ 6. ablations
    print("6/7 ablation studies ...")
    variants = {
        "Full method": {},
        "No oracle sampling": {"oracle_samples": 0},
        "No interpretability constraints": {"constrain": False},
        "Trained on true labels (not the network)": {"target": "truth"},
    }
    abl = []
    for label, kwargs in variants.items():
        for seed in seeds:
            s = fit_fuzzy_surrogate(data, bb, selected, seed=seed, **kwargs)
            ev = evaluate_surrogate(s, data, bb)["simplified"]
            drift = np.abs(s.full.centers_ - s.full.initial_centers_).max()
            order_ok = bool(np.all(np.diff(s.full.centers_, axis=1) > 0))
            abl.append({"variant": label, "seed": seed, **ev,
                        "max_term_drift": float(drift), "terms_stay_ordered": order_ok})
    abl = pd.DataFrame(abl)
    abl.to_csv(OUT / "ablations.csv", index=False)
    abl_summary = (abl.groupby("variant", sort=False)
                   .agg(fidelity=("fidelity", "mean"), accuracy=("accuracy", "mean"),
                        rules=("rules", "mean"), conditions=("conditions", "mean"),
                        max_term_drift=("max_term_drift", "mean"),
                        ordered=("terms_stay_ordered", "mean"))
                   .reset_index())

    feat_rows = []
    for k in (2, 3, 4, 5, 6):
        sel_k = top_features(importance, data.feature_names, k)
        for seed in seeds:
            ev = evaluate_surrogate(fit_fuzzy_surrogate(data, bb, sel_k, seed=seed), data, bb)["simplified"]
            feat_rows.append({"features": k, "grid_rules": config.N_TERMS ** k, **ev})
    feat = (pd.DataFrame(feat_rows).groupby("features")
            .agg(grid_rules=("grid_rules", "first"), rules=("rules", "mean"),
                 conditions=("conditions", "mean"), fidelity=("fidelity", "mean"),
                 accuracy=("accuracy", "mean"))
            .reset_index())
    feat.to_csv(OUT / "feature_count.csv", index=False)

    # -------------------------------------------------------- 7. summary
    print("7/7 writing summary ...")
    tree = fit_tree_surrogate(Xtr, bb.predict(data.X_train), report["main"]["simplified"]["rules"])
    main = report["main"]
    metrics_tbl = pd.DataFrame([
        {"model": "Neural network (black box)", **{k: nn_test[k] for k in ("accuracy", "f1", "roc_auc")}},
        {"model": "Logistic regression (reference)", **{k: lr_test[k] for k in ("accuracy", "f1", "roc_auc")}},
        {"model": "Fuzzy explainer, full",
         **{k: v for k, v in classification_metrics(data.y_test, sur.full.predict_proba(Xt)).items()
            if k in ("accuracy", "f1", "roc_auc")}},
        {"model": "Fuzzy explainer, simplified",
         **{k: v for k, v in classification_metrics(data.y_test, model.predict_proba(Xt)).items()
            if k in ("accuracy", "f1", "roc_auc")}},
    ])
    fid_tbl = pd.DataFrame([{"stage": stage, **{k: main[stage][k] for k in (
        "rules", "conditions", "avg_rule_length", "fidelity", "fidelity_boundary",
        "fidelity_confident", "prob_mae")}} for stage in ("full", "simplified")])
    best_curve = curve.copy()
    best_curve = best_curve[["method", "target_rules", "rules_mean", "conditions_mean",
                             "fidelity_mean", "fidelity_std", "accuracy_mean"]]
    grid_rows = sweep[sweep["method"].str.startswith("ANFIS grid")][["method", "rules", "conditions", "fidelity"]]

    summary = f"""# Results summary

_Generated by `scripts/run_experiments.py` ({'quick' if quick else 'full'} run,
{len(seeds)} seed(s), {time.time() - t0:.0f} s). Test set: {len(data.y_test)} patients
never used for training, tuning or pruning._

## 1. Black-box neural network

{bb.describe()}. Chosen by 5-fold cross-validation (top 5 of {len(cv_table)} settings):

{md_table(cv_table.head(5))}

Test-set performance against the true diagnosis:

{md_table(metrics_tbl)}

The network is about as accurate as logistic regression - typical for this small
tabular dataset. That does not matter for this project: the goal is to *explain*
whatever the network does, not to beat it.

## 2. What the network relies on

![importance](figures/feature_importance.png)

{md_table(importance[["feature", "importance", "share"]])}

The fuzzy system uses the top {config.N_FEATURES}: **{', '.join(names)}**.

## 3. Fidelity of the fuzzy explainer (test set)

{md_table(fid_tbl)}

- *fidelity* = share of test patients where fuzzy decision = network decision
- *boundary / confident* = fidelity where the network's probability is within / outside 0.35-0.65
- *prob_mae* = mean |fuzzy risk - network probability|

The rules themselves are in [rules.md](rules.md); worked examples in
[case_studies.md](case_studies.md).

![terms](figures/membership_functions.png)

## 4. Fidelity vs interpretability trade-off

![tradeoff](figures/tradeoff.png)

{md_table(best_curve)}

Classic grid-partition ANFIS on the same {len(selected)} inputs (every combination of terms):

{md_table(grid_rows)}

## 5. Ablation studies (simplified rule base, mean over seeds)

{md_table(abl_summary)}

- **No oracle sampling** - trained only on the {len(data.y_train)} real training patients.
- **No interpretability constraints** - terms may move anywhere and swap order,
  so "Low" may no longer be low (see *max_term_drift*, as a fraction of the feature range).
- **Trained on true labels** - fidelity to the network drops: imitating the network
  and solving the task are different goals.

## 6. Number of input features

{md_table(feat, ".3f")}

*grid_rules* is how many rules a full grid ANFIS would need ({config.N_TERMS}^k);
FCM initialisation plus pruning keeps the rule base small regardless of k.

## 7. Decision-tree surrogate at the same size

{tree_complexity(tree)['rules']} leaves, test fidelity
{fidelity(tree.predict(Xt), nn_lab):.3f}:

```
{tree_rules_text(tree, names)}```
"""
    (OUT / "summary.md").write_text(summary)
    report["ablations"] = abl_summary.to_dict(orient="records")
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"done in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="one seed instead of five")
    main(parser.parse_args().quick)
