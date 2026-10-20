# Quick Revision: Problem and Solution in Two Pages

Full explanations: [Project guide](guide/README.md).

---

## The problem (in five lines)

1. A **neural network** predicts diabetes from 8 measurements, but it cannot say *why*.
2. In medicine, unexplained decisions are hard to trust or check.
3. **Fuzzy IF-THEN rules** are readable, but normally have to be written by experts.
4. **Goal:** automatically build a *small* set of readable fuzzy rules that makes
   the **same decisions as the network** (high *fidelity*).
5. Also measure the **trade-off**: fewer rules are easier to read but usually less faithful.

> We are **not** trying to predict diabetes better than the network.
> We are trying to **explain the network** faithfully.

## The solution (in seven steps)

| # | Step | One line |
|---|---|---|
| 1 | Clean the data | Pima dataset, 768 patients; zeros in 5 columns = missing → training median; split 460 / 154 / 154 |
| 2 | Train the black box | Neural network 8 → 8 → 1 (70.1 % accuracy); from now on we only *ask it questions* |
| 3 | Pick inputs | Shuffle each input and see how much the network's answer moves → Glucose, BMI, Age, Pregnancies |
| 4 | Ask for more examples | 2,000 made-up patients near real ones, labelled by the network ("oracle sampling") |
| 5 | Propose rules | **Fuzzy C-Means** clustering → 8 rules in words (Low / Medium / High) |
| 6 | Tune the rules | **ANFIS** hybrid learning: least squares for the rule outputs, gradient descent for the word curves, with **constraints** so the words keep their meaning |
| 7 | Simplify | **Pruning** deletes conditions and rules while fidelity stays within 1 point; then fine-tune |

## The ANFIS in one picture

```
inputs → [1] how Low/Medium/High is each value? → [2] multiply per rule (AND)
       → [3] share of each rule (divide by the total) → [4] share × rule output
       → [5] add them up = risk  (≥ 0.5 → diabetic)
```

## The final rules (4 rules, 12 conditions)

| | IF | THEN risk |
|---|---|---|
| R1 | Glucose Low AND BMI Low AND Age Low AND Pregnancies Low | 0.03 |
| R2 | Glucose Medium AND Pregnancies Medium | 0.35 |
| R3 | Glucose Medium AND BMI High AND Age Medium | 0.85 |
| R4 | Glucose High AND BMI Medium AND Age Medium | 0.85 |

## Numbers to remember

| | |
|---|---|
| Fidelity, main model (4 rules) | **92.2 %** (142 of 154 unseen patients) |
| Fidelity, average of 5 other seeds | **94.8 % ± 1.1** (≈ 5 rules, 13 conditions) |
| Decision tree, same size | 87.7 % (higher than the tree at every size, but not significant alone: p = 0.21) |
| Constant answer (floor) | 67.5 % |
| Before pruning | 8 rules, 32 conditions |
| Network accuracy | 70.1 % (logistic regression 70.8 %; always "no" 64.9 %) |
| Trained on true labels instead | 88.2 % fidelity → imitating the network matters |
| Weak matches | 15 % of patients (no rule fits well; flagged) |

## Worked example in one line

Patient #152 (glucose 184, BMI 37, age 31): network **0.84**, rules **0.84**;
98 % of the decision comes from **R4** (Glucose High 1.00 · BMI Medium 1.00 · Age Medium 0.93).

## Five things never to get wrong

1. **Fidelity ≠ accuracy.** Fidelity = agreement with the network.
2. **Membership ≠ probability.** 0.7 "Warm" is a degree of fit, not a chance.
3. **Sugeno, not Mamdani:** THEN-parts are numbers; a weighted average replaces defuzzification.
4. **Trained on the network's answers**, not the true diagnoses.
5. **Not novel:** established techniques, carefully combined and evaluated.

## Honest limitations (say them first)

One small dataset · small test set · network ≈ logistic regression · some terms
overlap (BMI 37 is both 0.52 "Low" and 0.53 "High") · interpretability measured by
size, not by people · no clinician review · the tree baseline had less training data.

## Run it

```
pip install -r requirements.txt
python -m pytest            # 26 passed
python -m webapp            # live lab → http://127.0.0.1:8000
```
