# Chapter 9 · Training, Optimisation and Evaluation

[← Chapter 8](08-code-walkthrough.md) · [Guide index](README.md) · [Next: Traceability →](10-traceability.md)

## 9.1 Training the black box

| Step | What happens | Evidence |
|---|---|---|
| Initialisation | weights and biases drawn uniformly from $\pm\sqrt{6/(\text{fan-in}+\text{fan-out})}$ (Glorot), seed 42 | ✅ scikit-learn source, `random_state=42` |
| Hyperparameter selection | 16 settings (4 sizes × 4 α), 5-fold stratified CV on the 460 training patients, scored by ROC-AUC; best = 8 neurons, α = 1.0 (CV AUC 0.826 ± 0.017) | ✅ `results/summary.md` §1 |
| Epochs | mini-batches of 200, shuffled each epoch, Adam with learning rate 0.001 | ✅ |
| Loss | cross-entropy + L2 penalty (§6.18) | ✅ |
| Stopping | training loss improves by less than $10^{-4}$ for more than 10 epochs → stop (here at epoch 321; maximum 500) | ✅ |
| Inference | `predict_proba` → P(diabetic); `predict` → threshold 0.5 | ✅ |

Why no validation-based early stopping for the network? It was tried during
development: its internal hold-out was 15 % of 460 ≈ 70 patients, and results
varied a lot between runs. 🔎 This rationale is recorded in the `make_mlp`
docstring; the comparison runs themselves are not kept in the repository.

## 9.2 Training the fuzzy explainer

| Step | What happens | Evidence |
|---|---|---|
| Training pool | 460 real + 2,000 synthetic patients, 4 inputs, target = network probability | ✅ `pipeline.py` |
| Validation pool | 154 real + 1,000 synthetic, target = network probability | ✅ |
| Initialisation | terms at the 10/50/90 % quantiles (§6.2); rules from FCM with 8 clusters (§6.5) | ✅ |
| Each epoch (300) | least squares for $p$ → MSE → validation MSE → gradients for $c, \sigma$ → Adam step (learning rate 0.01) → constraints | ✅ `anfis.py::_train` |
| Stopping | fixed 300 epochs, then **restore the epoch with the lowest validation MSE** (epoch 21 for the main model) | ✅ |
| Simplification | greedy pruning with two checks (§6.15) | ✅ |
| Fine-tuning | 100 more epochs, same procedure; best validation epoch restored again | ✅ |
| Inference | `predict_proba` (clipped to [0, 1]), `predict` (≥ 0.5) | ✅ |

**What is learned vs fixed:**

| Learned from data | Fixed by us (hyperparameters) |
|---|---|
| network weights (81) | network size, α (chosen by CV) |
| term centres and widths (24 numbers) | number of terms (3), their names, initial positions rule |
| rule outputs $p_r$ (4 numbers) | number of FCM clusters (8), fuzzifier $m = 2$ |
| **which conditions and rules survive** (via pruning) | constraint limits (0.1 drift, 0.5 gap ratio, 0.5–2× width) |
| initial rule structure (via FCM) | learning rate 0.01, ridge λ = 1, epochs 300 + 100 |
| | pruning tolerance 1 pt, error growth 25 % |
| | inputs used: top 4 by importance |

## 9.3 Metrics: formulas, meaning, limits

Let TP, FP, TN, FN be true/false positives/negatives, with "positive" = diabetic.

| Metric | Formula | Meaning | Limitation |
|---|---|---|---|
| **Accuracy** | $\frac{TP+TN}{\text{all}}$ | share of correct decisions | misleading with imbalance: "always non-diabetic" scores 64.9 % here |
| **Precision** | $\frac{TP}{TP+FP}$ | of those called diabetic, how many are | ignores missed cases |
| **Recall** (sensitivity) | $\frac{TP}{TP+FN}$ | of the diabetic, how many were found | ignores false alarms |
| **F1** | $\frac{2PR}{P+R}$ | balance of precision and recall | one number hides the trade-off |
| **ROC-AUC** | area under the true-positive vs false-positive curve over all thresholds | probability that a random diabetic patient gets a higher score than a random non-diabetic one | says nothing about the chosen threshold |
| **Fidelity** | $\frac{1}{\lvert T\rvert}\sum_{n\in T}\mathbb 1[\hat d_{FIS}(x_n) = \hat d_{NN}(x_n)]$ | share of test patients where the explainer's decision = the network's | a constant answer already gets 67.5 % here |
| **Boundary / confident fidelity** | fidelity on patients with network probability within / outside 0.35–0.65 | where disagreements happen | small groups (35 boundary patients) |
| **Probability MAE** | $\frac{1}{\lvert T\rvert}\sum \lvert y_{FIS} - f_{NN}\rvert$ | average gap in risk | – |
| **Rules / conditions / avg rule length** | counts from the rule matrix | size (interpretability proxy) | size ≠ understanding |
| **Term drift / ordering** | max $\lvert c - c^0\rvert$; are centres still sorted? | did the words keep their meaning? | a proxy too |
| **Weak-match share** | share with $\max_r w_r < 0.2$ | how often no rule fits | threshold 0.2 is our choice |

MSE ($\frac1N\sum(y-t)^2$) is the ANFIS **training loss**, not a reported result.
RMSE and MAE against the true label are not used: the task is classification.

### Worked metric example (real, test set)

Network vs truth: TN = 79, FP = 21, FN = 25, TP = 29.
- Accuracy = (79 + 29)/154 = **0.701**
- Precision = 29/50 = **0.58**
- Recall = 29/54 = **0.537**
- F1 = 2(0.58 × 0.537)/(0.58 + 0.537) = **0.558** ✅

Explainer vs network (rows = network, columns = explainer):

| | explainer: non-diabetic | explainer: diabetic |
|---|---|---|
| **network: non-diabetic** | 96 | 8 |
| **network: diabetic** | 4 | 46 |

Fidelity = (96 + 46)/154 = **0.922**: 12 disagreements, 8 of them where the
explainer is more alarmed than the network. ✅

## 9.4 Baselines: what each comparison tells you

| Baseline | Value | What beating it shows |
|---|---|---|
| Always "non-diabetic" (accuracy) | 64.9 % | the network learned something (70.1 %), but not a lot |
| Logistic regression (accuracy) | 70.8 % | the network is **not** better than a linear model on this data |
| Constant explainer (fidelity) | 67.5 % | our explainer captures real structure (92.2 %) |
| Decision-tree surrogate, same size | 87.7 % (4 leaves) | fuzzy rules are more faithful at equal size, **but see §9.6** |
| Grid ANFIS (81 rules) | 94.2 % | our 4–5 rules reach about the same fidelity with 5–25× fewer conditions |
| Explainer trained on true labels | 88.2 % | imitating the network matters for fidelity |

## 9.5 What the results demonstrate, and what they do not

**They demonstrate** ✅
- A 4-rule fuzzy system can reproduce this network's decisions on 92 % of unseen
  patients; across five other seeds the average is 94.8 % ± 1.1 with about 5 rules.
- Pruning roughly halves the rule base for about 0.3 points of fidelity (seed average).
- The constraints keep terms ordered and close to their starting positions
  without costing fidelity.
- Each prediction can be decomposed exactly into rule contributions.

**They do not demonstrate** ❌
- That the network or the explainer is a good diabetes predictor (both are at
  the logistic-regression level).
- That people understand or trust the rules (no user study).
- That the method works on other datasets or larger networks.
- That the fuzzy explainer is *statistically significantly* more faithful than
  the tree (next section).

## 9.6 Statistical caution

- One test patient = **0.65 percentage points**. Differences of 1–2 points are 1–3 patients.
- **McNemar test**, explainer vs the 4-leaf tree, on the same 154 patients: the
  explainer alone agreed with the network on 15 patients, the tree alone on 8;
  exact two-sided **p = 0.21**. The difference is *consistent* across all nine
  sizes in the sweep, but **not significant** in any single comparison at this
  sample size. ✅ (computed for this guide with `scipy.stats.binomtest`; not part
  of `run_experiments.py`)
- The tree was trained on fewer points (460 real) with hard labels, while ANFIS
  had 2,460 points with soft targets. Without synthetic data, ANFIS still scores
  93.5 % vs the 5-leaf tree's 90.3 % at about 11–12 conditions, but the tree also
  lacks soft targets. A fairer comparison would give the tree the same pool.
  ([Chapter 11](11-results-critique.md))

## 9.7 Reproducibility

- Every random step takes a seed: data split 42, network 42, FCM and oracle
  sampling = the run's seed (42 for the main model, 0–4 for sweeps). ✅
- `python scripts/run_experiments.py` regenerates every number. Running it twice
  gave byte-identical figures. ✅
- On a different library version (Python 3.10, scikit-learn 1.7) the lab
  reproduced the same main-model predictions (0.84 / 0.84 for patient #152). ✅
- Not reproducible from the repository: the development comparisons quoted in
  docstrings (early stopping for the network; pruning without the
  probability-error check). ❓

[Next: Chapter 10 · Traceability →](10-traceability.md)
