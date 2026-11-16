# Chapter 10 · Connecting the Implementation to the Original Problem

[← Chapter 9](09-training-evaluation.md) · [Guide index](README.md) · [Next: Results & critique →](11-results-critique.md)

**Central question:** *how exactly does what we built begin to address the
original problem, and what is the evidence?*

## 10.1 Traceability table

| Original problem or challenge | Why it matters | Proposed mechanism | Actual implementation | Evidence of progress | Remaining limitation |
|---|---|---|---|---|---|
| The network's decisions are opaque | No trust, audit or appeal | Global fuzzy surrogate | `pipeline.fit_fuzzy_surrogate`: ANFIS trained on `blackbox.predict_proba` | 4 readable rules; per-patient explanations (`results/case_studies.md`) | Explains the *surrogate*; faithful on 92 % of patients, not 100 % |
| The surrogate must reflect the network, not the disease | Otherwise it explains a different model | Train on the network's probabilities | `t_fit = blackbox.predict_proba(...)` | Fidelity 94.8 % vs 88.2 % when trained on true labels (5-seed ablation) | – |
| Fidelity must be measured | The proposal's success criterion | Agreement on unseen data | `metrics.fidelity` on 154 test patients | 92.2 % main model; 94.8 % ± 1.1 average; 67.5 % constant baseline | Small test set; no significance test in the main script |
| Rules must come from data, not experts | Knowledge acquisition | Clustering-based rule generation | `rulebase.fcm_rules` over shared terms | 8 FCM clusters → 8 rules; the grid would need 81 | Number of clusters set by hand |
| Membership functions must be learned | Hand-tuned terms do not fit the network | Gradient-based premise tuning (hybrid learning) | `ANFIS._premise_gradients` + `Adam`; least squares for $p$ | Training MSE 0.019 → 0.006; gradient verified by test | Early stopping chose epoch 21; most later learning did not help |
| Learning can destroy word meaning | Unreadable or false rules | Constraints on drift, order, gap, width | `ANFIS._apply_constraints` | Drift held at 0.10 vs 0.50 unconstrained; ordering 100 % vs 60 % | Widths may double, so some terms overlap heavily (BMI 37 is Low 0.52 *and* High 0.53) |
| The rule base must be small | Readability | Pruning with fidelity and error checks | `simplify.simplify` + `fine_tune` | 32 → 12 conditions (main model); 26.4 → 12.8 average for 0.3 points | Greedy, not optimal |
| The trade-off must be quantified | The proposal's third requirement | Sweep over sizes × seeds | `run_experiments.py` stage 5; `tradeoff.png` | 9 sizes × 5 seeds; tree and grid references | One dataset, one network |
| Explanations should be traceable per decision | Case studies (objective 4) | Exact decomposition $y = \sum \bar w_r p_r$ | `explain.explain_instance` | Contributions sum to the output (tested) | – |
| Explanations should not overclaim | Clinical safety | Report the best rule's raw firing | `match_strength`, `WEAK_MATCH = 0.2` | 15 % of test patients flagged | Threshold 0.2 chosen by hand |
| Rules should make domain sense | Validation (objective 4) | Expert review | ❌ only the authors' judgement | Rules match common risk factors (high glucose, high BMI → high risk) 🔎 | No clinician involved |
| Practical usability vs the raw black box | Objective 5 (original) | User study | ❌ not done | – | Replaced by the tree baseline and ablations |
| Show it is better than a simpler explainer | Value of the fuzzy approach | Same-size decision-tree surrogate | `surrogates.fit_tree_surrogate` | Higher at all 9 sizes (e.g. 92.2 % vs 87.7 %) | Not significant on its own (McNemar p = 0.21); the tree had less data |

## 10.2 The same story in prose

**Intended:** explain a black-box network with a small, faithful, readable fuzzy
rule base, and quantify the trade-off.

**Designed mechanism:** a pedagogical surrogate. Query the network; learn fuzzy
rules that reproduce its decisions; keep the words meaningful; keep the rule
base small; measure fidelity and size.

**Implemented mechanism:** exactly that, with three engineering additions that
came from problems we observed while building it:
1. *constraints*, because unconstrained terms drifted. In a development run,
   "BMI Low" ended up centred at about 44 kg/m². That run is not preserved in the
   repository ❓, but the unconstrained ablation reproduces the effect (drift
   0.50, order broken in 2 of 5 runs) ✅;
2. *shoulder terms*, because a plain Gaussian made glucose 200 "less High" than
   155 (commit "Use shoulder terms and a relative minimum gap between terms") ✅;
3. the *probability-error check* in pruning, because pruning with fidelity alone
   pushed patients towards the 0.5 threshold (commit "Tighten interpretability:
   width limits and error guard when pruning") ✅.

The project grew by meeting real problems, which you can describe honestly in the viva.

**Observed results:** 92.2 % fidelity with 4 rules on unseen patients (94.8 %
average over five other seeds), consistently above same-size decision trees,
with constraints that cost no fidelity.

**Justified conclusions:**
- ✅ *For this network and dataset*, a handful of constrained fuzzy rules can
  reproduce most of the network's decisions and explain each one exactly.
- ✅ Imitating the network (not the truth), synthetic oracle data and the
  constraints each help or cost nothing (ablations).
- ❌ We **cannot** conclude that fuzzy surrogates beat trees in general, that the
  rules are clinically valid, or that people find them useful.

## 10.3 Partial vs complete

| Part of the original problem | Status |
|---|---|
| Faithful approximation | ✅ largely addressed (92–95 %) |
| Small rule base | ✅ addressed |
| Semantically interpretable | ⚠️ partly: ordered and anchored terms, but some overlap heavily, and no human evaluation |
| Trade-off quantified | ✅ addressed |
| Case studies | ✅ addressed |
| Practical usability vs the black box | ❌ not addressed |

[Next: Chapter 11 · Results & critique →](11-results-critique.md)
