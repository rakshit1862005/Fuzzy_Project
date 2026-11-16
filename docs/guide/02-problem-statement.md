# Chapter 2 · The Problem Statement and Its Real Meaning

[← Chapter 1](01-overview.md) · [Guide index](README.md) · [Next: Background →](03-background.md)

There are two versions of the problem statement in this project:

1. **The original proposal** written by the team at the start (the text that
   begins "Glass-Box Insight: Fuzzy Rule Extraction for Explaining Black-Box
   Neural Network Decisions" and offers the alternative titles "FIDEL" and
   "Neuro-Fuzzy XAI"). It is not stored in the repository; it was supplied in
   the conversation that produced the code.
2. **The revised statement** in [`docs/PROBLEM_STATEMENT.md`](../PROBLEM_STATEMENT.md),
   which shortens the motivation, fixes the dataset (Pima diabetes) and adds
   related work and references.

This chapter walks through the **original** text, because that is what the
project was judged against at the proposal stage, and notes where the revised
version and the code differ from it.

---

## 2.1 Line-by-line reading

### "Modern neural networks achieve high predictive accuracy … but they operate as black boxes"

- **Ordinary meaning:** these programs are often right, but you cannot see inside them.
- **Technical meaning:** a neural network's output is a composition of weighted
  sums and non-linear functions; its "knowledge" is spread across many weights,
  none of which corresponds to a human concept. Our network has 81 weights and
  biases (8×8 + 8 + 8×1 + 1). ✅ (`results/summary.md`, §1)
- **Underlying challenge:** *opacity*. Even when every number is visible, the
  reasoning is not, because no single weight means "high glucose".
- **Why it exists:** networks are optimised only to reduce prediction error.
  Nothing in their training encourages human-readable structure.
- **In our domain:** the network says "0.84 probability of diabetes" for test
  patient #152 but cannot say that it was mainly the glucose of 184 mg/dL.

### "Regulatory frameworks (e.g., GDPR's 'right to explanation') … increasingly demand … human-interpretable justification"

- **Ordinary meaning:** laws and hospitals want reasons, not just answers.
- **Technical meaning:** explainability is a *requirement* in some deployments,
  not a nice extra. 🔎 Whether GDPR creates a strict legal "right to
  explanation" is debated by legal scholars. In the viva, say "GDPR and the EU
  AI Act push towards explainable automated decisions" rather than claiming a
  specific legal right.
- **Relevance:** it motivates the project; it is not something the project tests.

### "fuzzy logic systems, built on human-readable IF-THEN rules, are inherently interpretable but typically lack the capacity to learn complex decision boundaries directly from data"

- **Ordinary meaning:** fuzzy rules read like sentences, but writing them by hand
  does not scale, and simple rule sets cannot capture complicated patterns.
- **Technical meaning:** a classical fuzzy system needs a human to choose the
  membership functions (what "High" means) and the rules. A small rule base is a
  coarse approximation of a complicated function.
- **Underlying challenge:** *knowledge acquisition* (where do the rules come
  from?) and *expressiveness vs readability*.
- **How the project addresses it:** rules are generated from data by Fuzzy
  C-Means, and their parameters are learned by ANFIS. ✅

### "a trained black-box neural network is treated as an oracle, and a fuzzy inference system is trained to approximate its decision behavior"

- **Ordinary meaning:** we ask the network questions and teach the fuzzy system
  to give the same answers.
- **Technical meaning:** this is a **global surrogate model**. The training target
  is the network's output $f_{NN}(x)$, not the true label $y$. In the code the
  target is the network's *probability*, a "soft" target. ✅
  (`glassbox/pipeline.py`, `fit_fuzzy_surrogate`, `t_fit = blackbox.predict_proba(...)`)
- **Underlying challenge:** a surrogate is only useful if it is *faithful*;
  otherwise it explains a different model.

### "The resulting fuzzy model does not aim to outperform the neural network — it aims to faithfully explain it"

- This sentence defines the success criterion. Judge the project by **fidelity**
  (agreement with the network), not by accuracy against the true diagnosis.
  Many examiners will initially ask about accuracy; this sentence is your answer.

### Problem definition: "construct a Fuzzy Inference System $f_{FIS}: X \to Y$ … such that"

1. **"$f_{FIS}$ closely approximates the behavior of $f_{NN}$ (measured via a fidelity metric)"**:
   implemented as the share of test patients on which both make the same
   decision. ✅ (`glassbox/metrics.py`, `fidelity`)
2. **"The resulting rule base remains small and semantically interpretable"**:
   *small* is measured by the number of rules and conditions ✅. *Semantically
   interpretable* is encouraged by constraints on the terms ✅, but was **not
   tested with people** ❓ (no user study exists).
3. **"the inherent trade-off between fidelity … and interpretability … is explicitly quantified"**:
   implemented as a sweep over rule-base sizes with 5 random seeds, plotted in
   `results/figures/tradeoff.png`. ✅

### Objectives 1–5

| # | Original objective | What was done | Status |
|---|---|---|---|
| 1 | "Train a feedforward neural network on a real-world tabular dataset (e.g., credit risk or medical diagnosis data) to achieve strong predictive performance" | MLP on Pima diabetes, size and L2 penalty chosen by 5-fold cross-validation. Test accuracy 70.1 %, ROC-AUC 0.80. | ✅ done; "strong" is **only partly met**: logistic regression scores about the same (70.8 %, 0.82) and always answering "non-diabetic" scores 64.9 %. |
| 2 | "Design and implement a fuzzy rule extraction pipeline (via Fuzzy C-Means clustering and/or ANFIS)" | Both: FCM proposes rules, ANFIS tunes them. | ✅ |
| 3 | "Define and compute quantitative metrics for fidelity … and interpretability … and generate a … trade-off analysis" | Fidelity (overall, near the boundary, far from it), probability gap, rules, conditions, rule length; sweep over 9 sizes × 5 seeds. | ✅ |
| 4 | "Interpret the extracted fuzzy rules in domain-meaningful terms and validate them through individual case studies" | Rules printed with units (`results/rules.md`); five case studies (`results/case_studies.md`). | ✅ case studies; ❓ validation "against domain intuition" was done by the authors' judgement, **not by a clinician**. |
| 5 | "Compare the explainability framework's practical usability against the raw black-box model" | Not measurable without users. The revised statement replaces it with a comparison against a decision-tree surrogate plus ablation studies. | ⚠️ **changed** (see §2.5) |

### Methodology bullets 1–5

All five are implemented: dataset selection and preprocessing (`data.py`),
black-box training and tuning (`blackbox.py`), fuzzy system construction with
gradient-based membership tuning (`fcm.py`, `rulebase.py`, `anfis.py`),
evaluation and trade-off curve (`metrics.py`, `scripts/run_experiments.py`),
and case studies (`explain.py`). ✅

### "Relevance to Course Learning Outcomes"

The proposal claims coverage of neural networks (Units 1–2), fuzzy sets and
rule-based systems (Units 4–5) and ANFIS. ✅ All three appear in the code. The
unit numbers come from the proposal; ❓ we cannot check them against the
syllabus.

---

## 2.2 Four levels of "the problem"

| Level | Statement |
|---|---|
| **Real-world problem** | Clinicians (or auditors) are asked to trust a model's diabetes-risk predictions without being told why. |
| **Mathematical problem** | Given a fixed function $f_{NN}: \mathbb{R}^8 \to [0,1]$ that we can evaluate but not interpret, find a function $f_{FIS}$ from a restricted, human-readable family (a few fuzzy IF-THEN rules) that matches $f_{NN}$'s decisions on new inputs as often as possible, while keeping the rule base small. |
| **Neuro-fuzzy research problem** | Data-driven fuzzy modelling tends to trade interpretability for accuracy. Learning tends to destroy the meaning of linguistic terms ("semantic drift"). Grid-shaped rule bases explode in size. How do we get faithful, small, meaningful rule bases automatically? |
| **Our project's specific problem** | On one tabular medical dataset, build and evaluate a pipeline (FCM rule proposals, constrained ANFIS, pruning) that produces a small, faithful fuzzy explanation of an MLP, and measure the fidelity–size trade-off against a decision-tree surrogate. |

## 2.3 Which neuro-fuzzy challenges genuinely motivate the method?

| Challenge | Applies? | How it shows up in our project |
|---|---|---|
| Learning non-linear relationships from data | ✅ yes | The network's decision surface is non-linear; ANFIS learns an approximation from data instead of hand-written rules. |
| Interpretability vs predictive performance (here: vs **fidelity**) | ✅ central | Fewer conditions usually means lower fidelity; measured in the trade-off sweep. |
| Choosing the number of rules and membership functions | ✅ yes | Grid ANFIS needs $3^k$ rules ($3^4 = 81$); we use FCM clusters to propose about 8 and pruning to remove what is not needed. The number of terms (3) is fixed by hand. |
| Fixed, hand-designed membership functions | ✅ yes | Our membership functions start at data quantiles and are then tuned by gradient descent. |
| Learning destroys the meaning of terms (semantic drift) | ✅ yes, and measured | Without constraints, terms drift by about half the input range and swap order in 2 of 5 runs (`results/summary.md`, §5). |
| Neural networks cannot give human-readable reasons | ✅ the motivation | The whole surrogate exists for this reason. |
| Representing uncertain or imprecise inputs | 🔎 partly | Fuzzy terms represent *vague category boundaries* ("Medium" glucose has no sharp edge). The project does **not** model measurement noise or probabilistic uncertainty in the inputs. |
| Traditional fuzzy systems cannot learn parameters | ✅ yes | That is why ANFIS (learnable fuzzy system) is used instead of a hand-built Mamdani system. |

## 2.4 A concrete example of the problem

Take **test patient #152**: 4 pregnancies, glucose 184 mg/dL, blood pressure
78 mm Hg, skin fold 39 mm, insulin 277 µU/mL, BMI 37.0, pedigree 0.264, age 31.
✅ (`data/diabetes.csv` via `load_dataset`; actual outcome: diabetic)

- **The network** says: P(diabetic) = **0.84**. Why? Its answer is the result of
  passing these eight numbers through 81 weights (one hidden layer of 8 ReLU
  neurons). Three of the eight neurons fire, with activations 1.64, 1.59 and
  0.41; the rest are zero ([Chapter 12](12-worked-example.md)). None of that is
  an explanation.
- **Our fuzzy explainer** says: risk = **0.84**, and 98 % of that decision comes
  from one rule:

  > **R4:** IF Glucose is High (membership 1.00) AND BMI is Medium (1.00) AND
  > Age is Medium (0.93) THEN risk = 0.85.

That is the gap the project closes: same answer, but with a reason a person
can check.

## 2.5 Where the proposal, the revised statement and the code differ

| Topic | Original proposal | Revised statement / code | Why |
|---|---|---|---|
| Domain | "credit risk or medical diagnosis" | Medical: Pima Indians Diabetes ✅ | All 8 features are numeric and meaningful, so they suit fuzzy terms. |
| "Strong predictive performance" | Expected | Network ≈ logistic regression (70.1 % vs 70.8 %) ✅ | Normal for this small dataset; the project's criterion is fidelity. |
| Objective 5 "practical usability" | Compare against the raw black box | Replaced by a decision-tree baseline and ablation studies ✅ | Usability needs a user study, which was not done. |
| "membership functions through gradient-based learning" | Yes | Yes, plus **constraints** that limit drift and width ✅ | Unconstrained learning destroyed the meaning of terms. |
| Rule extraction "via clustering of the input/activation space" | Possibly the network's hidden activations | Clustering of **inputs plus the network's output**, not hidden activations ✅ | Keeps the black-box assumption: no access to internals is needed. |
| Live demo | Not mentioned | Web lab `python -m webapp` ✅ | Added for the viva. |

## 2.6 What the project aims to do, and what is out of scope

**In scope:**
- Explain one trained network with a small fuzzy rule base on one dataset.
- Measure fidelity, size and the trade-off between them.
- Explain individual predictions exactly, and flag when an explanation is weak.
- Compare with a decision-tree surrogate; test each design choice by ablation.

**Out of scope** (do not claim these):
- Better diabetes prediction than the network or than other methods.
- Clinical validity, or a tool usable on real patients.
- Evidence that doctors understand or trust the rules (no user study).
- Explaining deep networks, images or text.
- Novelty: every building block (ANFIS, FCM, surrogate models, pruning,
  TREPAN-style sampling) already exists. The contribution is a careful
  combination and evaluation (see [Chapter 4](04-existing-approaches.md)).

[Next: Chapter 3 · Background concepts →](03-background.md)
