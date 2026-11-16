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
> The result is 4 to 5 short rules that agree with the network on about
> 92 to 95 % of unseen patients. Any single prediction can be traced back to the rules
> that produced it. We also measure the trade-off between faithfulness and
> simplicity and compare against a decision-tree explainer.

---

## 2. Who explains what (suggested split for a team of 4)

| Member | Part | Files | Must be able to explain |
|---|---|---|---|
| A | Data and neural network | `data.py`, `blackbox.py`, `feature_selection.py` | imputation without leakage, MLP + backprop + Adam, L2 regularisation, cross-validation, permutation importance |
| B | Fuzzy foundations | `membership.py`, `fcm.py`, `rulebase.py` | fuzzy sets vs crisp sets, Gaussian MFs, linguistic terms, FCM update equations, why grid partition explodes |
| C | ANFIS | `anfis.py`, `simplify.py` | the 5 layers, Takagi-Sugeno rules, hybrid learning, gradient derivation, constraints, pruning |
| D | Evaluation and demo | `metrics.py`, `explain.py`, `oracle.py`, `surrogates.py`, `run_experiments.py`, `webapp/` | fidelity vs accuracy, trade-off plot, ablations, decision-tree baseline, case studies, how the live lab streams training |

**Everyone** should know the pitch, the headline numbers, and questions 1 to 10 below.

---

## 3. Demo script (5-7 minutes)

Start the lab with `python -m webapp` before the viva; it needs no internet.

**Part 1 - train it live (Train tab).** The settings are the report's. Keep
*Slow motion* ticked and press **Train**. Narrate each panel as it fills (about
15 seconds in total):

1. *Neural network learns*: the loss falls epoch by epoch. That is
   backpropagation with Adam. Validation accuracy is shown next to it.
2. *What the network relies on*: shuffle one input and see how far the output
   moves. Glucose dominates; the four blue inputs go to the fuzzy system.
3. *Rule base*: Fuzzy C-Means proposes 8 rules, written in words.
4. *ANFIS learns*: the loss falls and the Low/Medium/High curves move a little
   away from their dashed starting positions. The constraints stop them moving
   further.
5. *Pruning*: conditions disappear one at a time (32 → 12), each edit keeping
   fidelity within 1 point.
6. *Result*: 92.2 % fidelity on 154 unseen patients with 4 rules. A decision tree
   of the same size gets 87.7 %.

**Part 2 - explain one patient (Explore tab).** It opens on **test patient #152**
(glucose 184, BMI 37, age 31). Both models say 0.84. Rule R4 ("Glucose is High
AND BMI is Medium AND Age is Medium") carries 98 % of the decision, and the
contributions add up exactly to the fuzzy risk. Then drag the **Glucose** slider
down (measured values):

| Glucose | Network | Fuzzy | Rule with the largest weight |
|---|---|---|---|
| 184 | 0.84 | 0.84 | R4 Glucose High (0.98) |
| 150 | 0.66 | 0.66 | R4 Glucose High (0.49), R2 Glucose Medium (0.30) |
| 130 | 0.46 | 0.51 | R2 Glucose Medium (0.49) - *the two disagree here* |
| 110 | 0.26 | 0.33 | R1 Glucose Low (0.41), R2 (0.38) |

At 130 the patient sits on the network's decision boundary, which is exactly
where the fuzzy system disagrees. That makes a good talking point. In the
membership charts the orange line is the patient: one value is partly "Medium"
and partly "High" at the same time, which is fuzzy logic.

Pick **test patient #108**: a weak-match warning appears. No rule describes a
67-year-old with very high glucose and low BMI well, and the system says so.

**Part 3 - let the examiner edit the rules (Edit rules tab).** Delete
"Glucose is Low" from R1: fidelity drops from 92.2 % to 89.0 % immediately,
because that condition was doing real work. Press *Reset to trained rules*.
Invite the examiner to delete or change any condition.

**Part 4 - let the examiner change a setting.** For example, untick *Keep terms
meaningful*, or set *Imitate* to *The true diagnosis*, untick *Slow motion*, and
train again (about 5 seconds). Open **Compare runs** to see both runs on the
fidelity-vs-size plot next to their decision trees.

---

## 4. Design decisions and why we made them

| Decision | Why |
|---|---|
| Pima diabetes dataset | All 8 inputs are numeric and medically meaningful, so they suit fuzzy terms. Small enough to train in seconds. |
| Median imputation of impossible zeros, fitted on training data only | A glucose of 0 is "not measured", not a real value. Fitting on training data only prevents test-set leakage. |
| Network tuned by 5-fold CV, no early stopping | Early stopping's hold-out was only about 70 patients and made results unstable. Stopping when the training loss levels off (at most 500 epochs; 321 in practice) plus L2 gave stable CV scores. |
| Surrogate learns the network's **probability** | It explains the network, not the disease. Probabilities also tell the surrogate how confident the network is. |
| Only the top 4 inputs (permutation importance) | Rules with 8 conditions are unreadable. The top 4 carry most of what the network uses. |
| Zero-order Takagi-Sugeno | "THEN risk = 0.85" is readable; a linear formula is not. We measured the alternative: first-order rules were *not* more faithful (94.3 % vs 94.8 %) and needed 18 numbers after THEN instead of 5. |
| Shared Low / Medium / High terms | "Glucose is High" means the same thing in every rule. |
| FCM instead of a grid of rules | A grid needs 3⁴ = 81 rules for 4 inputs; FCM proposes about 8. |
| Clustering inputs **and** the network output together | Clusters then group patients who look alike *and* get similar decisions. |
| Hybrid learning (LSE + gradient descent) | The output is linear in the rule outputs, so they have an exact optimum. Only the membership functions need gradient descent. This is Jang's original ANFIS. |
| Logarithms when multiplying memberships | A product of many small numbers underflows to 0; adding logs is the same maths, computed safely. |
| Shoulder terms at both ends | Without them, a glucose of 200 would be *less* "High" than 155, because a bell curve falls off again. |
| Constraints on centres and widths | Without them, terms drift by about half the input range and swap order (2 of 5 runs), so "Low" stops meaning low. Fidelity does not even improve (94.2 % vs 94.8 %), so the constraints cost nothing. |
| Oracle sampling | The network can label as many synthetic patients as we like, giving the surrogate more data (the TREPAN idea). Adds 1.3 points of fidelity in the ablation. |
| Match-strength flag | Normalised weights always sum to 1, so a rule can "win" even when it barely fits. Reporting the best raw firing strength exposes this. |
| Pruning with two checks + fine-tuning | Pruning roughly halves the rule base or more. The probability-error check stops it pushing patients to the 0.5 edge, where they flip on new data. |
| Decision-tree baseline at the same size | Answers "why not just use a tree?" with numbers: at 4 rules, tree 87.7 % vs fuzzy 92.2 %. |
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
  dropped four carry little weight. Permutation importance itself can mislead
  when inputs are correlated (Age and Pregnancies are), because shuffling one
  creates unrealistic patients.
- **Fidelity is not truth.** If the network is wrong, a faithful explanation
  faithfully explains a wrong decision. That is the point: it lets a doctor
  *see* that the reasoning is wrong.
- **Some patients have no good rule.** 15 % of test patients have a weak match
  (best rule fires below 0.2). Their explanations are flagged, not hidden.
- **Results vary with the random seed.** The main model (seed 42) has 92.2 %
  fidelity after pruning. Five *other* seeds (0 to 4) average 94.8 % ± 1.1 after
  pruning and 95.1 % ± 0.7 before. Compare before/after within the same seeds, not
  across them. That is why the trade-off study reports means and standard deviations.
- **The rules themselves change between seeds.** Different runs produce somewhat
  different rule bases with similar fidelity. Which one is "the" explanation is
  an open research question (explanation stability) that we did not solve.
- **Interpretability is measured by proxies** (rule and condition counts), not by
  asking doctors whether the rules help. That would need a user study.
- **Only one baseline explainer** (a decision tree). Standard XAI methods such as
  LIME, SHAP or Anchors were not compared.
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
would be a separate model that explains itself. In our ablation its fidelity
to the network drops from 94.8 % to 88.2 %.

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
point; (b) Low < Medium < High, and neighbours keep at least half their initial
distance; (c) widths stay between ½× and 2× their start. Without them, the
ablation shows terms drifting by about half the range and swapping order in 2 of
5 runs, with no gain in fidelity.

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
It is the standard global surrogate in XAI. Compare at equal size, never
unequal sizes:
- same number of rules (4): tree 87.7 %, our pruned rules 92.2 %;
- about the same number of conditions (~12): tree with 5 leaves 90.3 %, our
  pruned rules averaged over 5 seeds 94.8 %.

The fuzzy rules also give graded outputs instead of hard thresholds. Our
explanation for the gap (not proven): smooth, overlapping terms follow the
network's smooth decision surface better than axis-aligned cuts.

Two caveats to state: on 154 patients a single comparison is not statistically
significant (McNemar p = 0.21 at 4 rules; the lead is consistent across all nine
sizes), and the tree was trained on the 460 real patients' hard labels while
ANFIS used 2,460 points with soft targets.

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
Mostly. High glucose and high BMI lead to high risk, and low glucose to very
low risk. These match known risk factors (impaired glucose tolerance, obesity,
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
One full run (network plus fuzzy explainer, from the raw CSV) takes about 5
seconds; the lab's slow motion stretches it to about 15 seconds so it can be
followed. The full experiment suite with 5 seeds takes about 5 minutes on a
laptop (`--quick` for 1 seed).

**36. What are shoulder membership functions?**
The outer terms are flat beyond their centre: "Low" is 1 for every value below
its centre and "High" is 1 for every value above. A plain Gaussian "High" would
say a glucose of 200 is less High than 155, which is nonsense. Shoulders are
standard practice in fuzzy controllers.

**37. What is match strength and why do you need it?**
The normalised weights always sum to 1, so some rule always "wins", even for a
patient no rule fits. Match strength is the raw firing strength of the best
rule, $\max_r w_r$. Below 0.2 we flag the explanation as weak. This happens for
15 % of test patients, for example patient #108: a 67-year-old with very high
glucose but low BMI.

**38. Your rules don't mention BloodPressure or Insulin. Is that wrong?**
No. The network itself hardly uses them: together they account for under 8 % of
its permutation importance. The rules describe what the network relies on,
not textbook medicine.

**39. Why zero-order rules and not first-order (linear) THEN-parts?**
We tested both (ablation in `results/summary.md` §5). First-order rules were not
more faithful: 94.3 % vs 94.8 %, within the seed-to-seed spread. They needed 18
numbers after THEN instead of 5. They can also read strangely: in one run, a
rule's formula had risk *falling* as glucose rose. Zero-order gives the same
fidelity with outputs a doctor can read at a glance.

**40. Your main model has 92.2 % fidelity, but you also quote 94.8 %. Which is it?**
Both, for different things. 92.2 % is the single model we show (seed 42), which
happened to prune down to 4 rules. 94.8 % ± 1.1 is the average of five other
seeds for the same setting, which kept about 5 rules. Before pruning, those
same five seeds average 95.1 % ± 0.7. So pruning costs about 0.3 points on
average; our showcase run lost more because it pruned harder.

**41. Can you trust permutation importance?**
As a ranking of what the network uses, yes, approximately: it needs only the
network's outputs. Its known weakness is correlated inputs. Shuffling Age while
keeping Pregnancies creates unrealistic patients (a 22-year-old with 12
pregnancies) and can over- or under-state importance. Glucose's lead (44 %) is
large enough that the top choice is not in doubt.

**42. Is this a real research problem, or only a course exercise?**
The problem is real and active. The interpretability–accuracy trade-off is a
central theme of fuzzy modelling (Casillas et al., 2003; Gacto et al., 2011).
Fuzzy logic is argued to be a natural tool for explainable AI (Hagras, 2018),
and regulation such as GDPR and the EU AI Act pushes for explainable automated
decisions. Our project is a course-scale study of it. Every component we use
already exists; the contribution is combining and evaluating them carefully.
Turning it into research would need several datasets with statistical tests,
comparison with LIME, SHAP and Anchors, a user study with clinicians, and an
analysis of how stable the rules are across runs.

**43. Is the demo just showing precomputed results?**
No. Every press of *Train* trains a new network and a new fuzzy system from the
raw CSV, with the settings on screen, and streams each epoch and pruning step to
the browser as it happens. Predictions and rule edits are computed on demand by
the same Python models. With the default settings the lab reproduces the
report's main model exactly, because it calls the same code with the same seeds.

**44. How does the browser see training while it is running?**
The FastAPI server (`webapp/server.py`) runs the pipeline in a background thread
(`webapp/jobs.py`). Hooks in the training loops record an event after every
network epoch, every ANFIS epoch and every accepted pruning edit. The browser
holds one open connection (server-sent events) and draws each event as it
arrives. scikit-learn has no callback for MLP training, so `StreamingMLP` in
`blackbox.py` overrides the method scikit-learn calls once per epoch. The
training maths is untouched.

**45. Why not Streamlit?**
We started with Streamlit. It re-runs the whole script on every click, so it
could not show training while it ran, and every interaction redrew the page. A
small FastAPI server plus a plain HTML/JavaScript page can stream training,
answer slider moves in milliseconds and edit rules in place. It still has no
build step and no internet dependency.
