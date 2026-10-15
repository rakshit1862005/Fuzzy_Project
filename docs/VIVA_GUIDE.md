# Viva Guide

Everything you need to present and defend the project. Read
[METHODOLOGY.md](METHODOLOGY.md) for the equations and
[../results/summary.md](../results/summary.md) for the numbers.

---

## 1. The 2-minute pitch

> Neural networks predict well, but nobody can see why they decide what they
> decide. Fuzzy rule systems are the opposite: anyone can read "IF Glucose is
> High THEN risk is high", but they are weaker learners.
>
> We trained a neural network to predict diabetes on the Pima dataset and then
> treated it as a black box. We trained a fuzzy system, an ANFIS, to **copy the
> network's decisions**, not the true labels. So the rules describe *the
> network*.
>
> Fuzzy C-Means proposes the rules. ANFIS tunes them with hybrid learning:
> least squares for the rule outputs and gradient descent for the membership
> functions. Constraints keep "Low", "Medium" and "High" meaningful. Finally we
> prune every condition that is not needed.
>
> The result is a handful of short rules that agree with the network on about
> 94 % of unseen patients. Any single prediction can be traced back to the rules
> that produced it. We also measure the trade-off between faithfulness and
> simplicity and compare against a decision-tree explainer.

---

## 2. Who explains what (suggested split for a team of 4)

| Member | Part | Files | Must be able to explain |
|---|---|---|---|
| A | Data and neural network | `data.py`, `blackbox.py`, `feature_selection.py` | imputation without leakage, MLP + backprop + Adam, L2 regularisation, cross-validation, permutation importance |
| B | Fuzzy foundations | `membership.py`, `fcm.py`, `rulebase.py` | fuzzy sets vs crisp sets, Gaussian MFs, linguistic terms, FCM update equations, why grid partition explodes |
| C | ANFIS | `anfis.py`, `simplify.py` | the 5 layers, Takagi-Sugeno rules, hybrid learning, gradient derivation, constraints, pruning |
| D | Evaluation and demo | `metrics.py`, `explain.py`, `oracle.py`, `surrogates.py`, `run_experiments.py`, `app.py` | fidelity vs accuracy, trade-off plot, ablations, decision-tree baseline, case studies, live demo |

**Everyone** should know the pitch, the headline numbers, and questions 1 to 10 below.

---

## 3. Demo script (5 minutes)

1. `streamlit run app.py`, then open the **About** tab: what the black box is and
   what the explainer is.
2. **Test-set patient 0** - read the two predictions and point at "Do they agree?".
3. **Why this decision?** tab:
   - the bar chart shows which rules carried the decision;
   - in the table, show that the *contributions add up exactly to the fuzzy risk*.
4. **Linguistic terms** tab - the orange line is the patient. Show how a value is
   partly "Medium" and partly "High" at the same time. That is fuzzy logic.
5. Switch to **Enter values**. Raise Glucose step by step and watch the rule
   weights shift from "Glucose is Low" rules to "Glucose is High" rules.
6. **All rules** tab - the full rule base fits on one screen.
7. End on the trade-off plot in `results/figures/tradeoff.png`.

---

## 4. Design decisions and why we made them

| Decision | Why |
|---|---|
| Pima diabetes dataset | All 8 inputs are numeric and medically meaningful, so they suit fuzzy terms. Small enough to train in seconds. |
| Median imputation of impossible zeros, fitted on training data only | A glucose of 0 is "not measured", not a real value. Fitting on training data only prevents test-set leakage. |
| Network tuned by 5-fold CV, no early stopping | Early stopping's hold-out was only about 70 patients and made results unstable. Fixed epochs plus L2 gave stable CV scores. |
| Surrogate learns the network's **probability** | It explains the network, not the disease. Probabilities also tell the surrogate how confident the network is. |
| Only the top 4 inputs (permutation importance) | Rules with 8 conditions are unreadable. The top 4 carry most of what the network uses. |
| Zero-order Takagi-Sugeno | "THEN risk = 0.83" is readable; "THEN y = 0.3·Glucose − 1.2·BMI + …" is not. |
| Shared Low / Medium / High terms | "Glucose is High" means the same thing in every rule. |
| FCM instead of a grid of rules | A grid needs 3⁴ = 81 rules for 4 inputs; FCM proposes about 8. |
| Clustering inputs **and** the network output together | Clusters then group patients who look alike *and* get similar decisions. |
| Hybrid learning (LSE + gradient descent) | The output is linear in the rule outputs, so they have an exact optimum. Only the membership functions need gradient descent. This is Jang's original ANFIS. |
| Logarithms when multiplying memberships | A product of many small numbers underflows to 0; adding logs is the same maths, computed safely. |
| Constraints on centres and widths | Without them, training can move "Low BMI" to BMI 44 or swap Low and High. Fidelity rises slightly, but the words become lies (see ablation). |
| Oracle sampling | The network can label as many synthetic patients as we like, giving the surrogate more data (the TREPAN idea). |
| Pruning with two checks + fine-tuning | Pruning roughly halves the rule base or more. The probability-error check stops it pushing patients to the 0.5 edge, where they flip on new data. |
| Decision-tree baseline at the same size | Answers "why not just use a tree?" with numbers. |
| Implemented from scratch with NumPy | So every line can be explained in the viva. Gradients are verified numerically in the tests. |

---

## 5. Honest limitations (say these before the examiner does)

- **The network is not better than logistic regression on this dataset** (both
  about 70 % test accuracy). That is common for small tabular data. The project
  is about explaining the network, not beating a baseline.
- **The test set is small** (154 patients), so one patient is 0.65 percentage
  points of fidelity. That is why the trade-off study averages 5 random seeds.
- **The fuzzy system uses 4 of the 8 inputs.** If the network relied heavily on a
  dropped input, the explanation would miss it. Permutation importance shows the
  dropped four carry little weight.
- **Fidelity is not truth.** If the network is wrong, a faithful explanation
  faithfully explains a wrong decision. That is the point: it lets a doctor
  *see* that the reasoning is wrong.
- **Not a medical tool.** The data are from one population of 768 women.

---

## 6. Likely questions and answers

### Concept

**1. What is the difference between accuracy and fidelity?**
Accuracy compares a model with the true diagnosis. Fidelity compares the fuzzy
system with the *network*. An explainer needs high fidelity; its accuracy is
only as good as the network's.

**2. Why train the fuzzy system on the network's outputs and not the real labels?**
Because we want to explain the network. Trained on real labels, the fuzzy system
would be a separate model that explains itself. Our ablation shows its fidelity
to the network drops clearly.

**3. Why not just use the fuzzy system and throw the network away?**
In this project the network is the given model whose decisions need
explaining, for example one already deployed. In general the network can learn
boundaries a small rule base cannot. The explainer is a readable summary of it,
which is why fidelity is below 100 %.

**4. What is a fuzzy set?**
A set where membership is a degree between 0 and 1 instead of yes/no. Glucose
130 can be 0.6 "Medium" and 0.4 "High" at the same time.

**5. What is a membership function? Why Gaussian?**
It maps a value to its degree of membership. Gaussians are smooth, so they can
be tuned by gradient descent. Each needs only two parameters: centre and width.

**6. What is ANFIS?**
A Takagi-Sugeno fuzzy system written as a 5-layer network (fuzzification,
rule firing, normalisation, consequent, output), so its parameters can be
learned from data like a neural network's.

**7. Mamdani vs Takagi-Sugeno?**
Mamdani rules output a fuzzy set ("risk is High") and need defuzzification,
e.g. centroid. Sugeno rules output a number or formula, and the system output
is a weighted average, so no defuzzification step is needed. That makes Sugeno
systems differentiable and suitable for ANFIS. We use zero-order Sugeno
(constant outputs), so the outputs still read like "risk 0.83".

**8. Which t-norm do you use for AND?**
Product: $w_r = \prod_j \mu_j$. It is differentiable everywhere. Min is the
alternative but has a gradient for only one input at a time.

**9. Where is defuzzification?**
With Sugeno rules, layer 5 (the weighted average $y = \sum_r \bar w_r p_r$)
plays that role.

**10. Explain hybrid learning.**
In each epoch: (1) forward pass with the membership functions fixed: the output
is linear in the rule outputs $p$, so we solve for the best $p$ exactly by least
squares; (2) backward pass with $p$ fixed: gradient descent on the centres and
widths. It converges faster than pure gradient descent because half the
parameters are solved exactly.

### Neural network

**11. Describe your network.**
See `results/summary.md` §1: standardised inputs, one ReLU hidden layer chosen
by cross-validation, and a sigmoid output. Trained by backpropagation with
Adam on cross-entropy loss plus an L2 penalty.

**12. Why ReLU? Why sigmoid at the output?**
ReLU avoids vanishing gradients and is cheap. A sigmoid output gives a
probability for a binary class.

**13. How did you prevent overfitting?**
L2 weight penalty (alpha), choosing the size by 5-fold cross-validation, and a
small network for a small dataset.

**14. What is backpropagation?**
Applying the chain rule layer by layer from the loss back to every weight to
get the gradient, then updating the weights against the gradient.

**15. What does Adam do?**
Gradient descent with momentum (a running mean of gradients) and a per-parameter
step size (scaled by a running mean of squared gradients). We use it both for
the network and for the ANFIS premise parameters.

### Fuzzy C-Means and rules

**16. How does FCM differ from k-means?**
k-means gives each point one cluster; FCM gives each point a membership in every
cluster, summing to 1. The fuzzifier m (we use 2) controls how soft the clusters
are.

**17. How do clusters become rules?**
For each cluster centre and each input, we choose the term (Low/Medium/High)
with the highest membership at the centre. That sequence of terms is the
IF-part. Clusters that give the same sentence are merged.

**18. Why not a grid of all combinations?**
Rules grow as (terms)^(inputs): 81 for 4 inputs, 6561 for 8. Most would never
fire and nobody could read them. See the "Number of input features" table.

**19. How do you choose the number of rules?**
The number of FCM clusters sets an upper bound; pruning then removes what is not
needed. The trade-off plot shows fidelity for each size, and we choose a point
where adding more rules no longer helps.

### Training details

**20. Derive the gradient for a centre.**
$y = \sum_r \bar w_r p_r$ gives $\partial y/\partial\theta = \sum_r \bar w_r
(p_r - y)\,\partial \log w_r/\partial\theta$. For a Gaussian,
$\partial \log\mu/\partial c = (x-c)/\sigma^2$. Multiply by
$\partial L/\partial y = 2(y-t)/N$ and sum over patients. The test
`test_analytic_gradient_matches_numerical` checks this against finite
differences.

**21. What constraints do you apply and why?**
After every step: (a) centres stay within ±10 % of the range of their starting
point; (b) Low < Medium < High with a minimum gap; (c) widths stay between ½× and
2× their start. Without them, the ablation shows terms drifting by more than
half the range and swapping order. "Low" would no longer mean low.

**22. Why is the ridge term in least squares needed?**
A rule that almost never fires has a near-zero column, so plain least squares
could give it an arbitrary output. The ridge term pulls it towards the average
risk.

**23. What is oracle sampling? Isn't synthetic data cheating?**
We add noise to real training patients and ask the network for its
probability. It is not cheating because the target *is* the network: its
answer on any input is by definition correct for our purpose. Test patients
are never used.

**24. How does pruning work?**
Greedily try deleting each condition, re-fit the rule outputs, and keep the best
deletion if fidelity drops by at most 1 percentage point and the probability
error grows by at most 25 %. Repeat until nothing qualifies, then do the same
for whole rules, then fine-tune.

**25. Why the second (probability-error) check?**
Fidelity only counts flipped decisions. With fidelity alone, pruning moved many
patients close to 0.5. They still counted as agreeing on validation data but
flipped on test data, so test fidelity fell from 94 % to 92 %. The extra check
removed that loss.

### Evaluation

**26. How do you measure interpretability?**
Number of rules, total number of conditions, average rule length, and whether
terms keep their meaning (ordered, limited drift).

**27. What does the trade-off plot show?**
Fidelity (y) against rule-base size (x) for full ANFIS, simplified ANFIS and a
decision tree. The upper-left is best. Simplified ANFIS reaches similar fidelity
with far fewer conditions.

**28. Why compare with a decision tree?**
It is the standard global surrogate in XAI. At the same number of rules, our
fuzzy rules are as faithful or more faithful (see `summary.md` §4) and give
graded, smooth outputs instead of hard thresholds.

**29. Where does the fuzzy system disagree with the network?**
Mostly near the network's decision boundary (probability 0.35 to 0.65); see
"boundary" vs "confident" fidelity. That is where the network's surface bends
and a few rules cannot follow every bend.

**30. How do you explain one prediction?**
The output is $\sum_r \bar w_r p_r$, so each rule's contribution is
$\bar w_r p_r$. We list the rules with the largest weights, show how strongly
each of their conditions is satisfied, and the contributions add up exactly to
the output. See `results/case_studies.md`.

**31. Do the rules make medical sense?**
Yes. High glucose and high BMI lead to high risk, and low glucose to very low
risk. These match known risk factors (impaired glucose tolerance, obesity,
age). Show `results/rules.md`.

**32. How would you extend this?**
Use more datasets (e.g. credit risk); learn the number of terms per feature;
explain a deeper network; try Mamdani rules for an all-linguistic output; run
a user study on whether doctors find the rules useful.

**33. Why is test fidelity not 100 %?**
A few smooth rules cannot exactly match a network with 8 inputs, of which we
use 4. Some fidelity is the price of readability; that is the trade-off the
project measures.

**34. Is the result reproducible?**
Yes. Every random step takes a seed (`config.py`), and
`python scripts/run_experiments.py` regenerates every table and figure.

**35. How long does it take to run?**
Training the network plus the explainer takes about 30 seconds. The full
experiment suite with 5 seeds takes about 10 to 15 minutes on a laptop
(`--quick` for 1 seed).
