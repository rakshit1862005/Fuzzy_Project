# Chapter 13 · Viva Questions and Model Answers

[← Chapter 12](12-worked-example.md) · [Guide index](README.md) · [Next: Speaking guide →](14-speaking-guide.md)

Format: **Q** → *Short answer* (say this first) → **More detail** (for follow-ups)
→ 📎 the evidence to point at. ⭐ marks the most likely questions.

---

## Level 1 · Basic understanding

**1.1 ⭐ What is your project in one sentence?**
*We trained a neural network to predict diabetes, then trained a small
neuro-fuzzy rule system to imitate it, so that its decisions can be explained in
words.*
📎 [Chapter 1](01-overview.md)

**1.2 ⭐ What is the difference between accuracy and fidelity?**
*Accuracy compares a model with the true diagnosis; fidelity compares our
explainer with the network.*
**More detail:** our explainer is judged by fidelity (92.2 %). Its accuracy
(70.1 %) is about the network's, because it copies the network.
📎 `metrics.py::fidelity`

**1.3 What is a fuzzy set?**
*A set where membership is a degree between 0 and 1 instead of yes or no.* A
glucose of 130 mg/dL is "High" to degree 0.30 in our model.
📎 [Figure 3.1](figures/crisp_vs_fuzzy.png)

**1.4 Is a membership degree a probability?**
*No. It measures how well a value fits a word, not a chance.* BMI 37 has degrees
0.52, 1.00 and 0.53 for Low, Medium and High, which do not sum to 1.

**1.5 What is a membership function? Which do you use?**
*A function that gives the membership degree.* We use Gaussians: a bell for
Medium and half-bell "shoulders" for Low and High, so extreme values are fully
Low or High.
📎 `membership.py::term_offsets`

**1.6 What is a linguistic variable?**
*A variable described by words.* Glucose, with the terms Low, Medium and High.

**1.7 What is a fuzzy rule? Give one of yours.**
*IF-THEN with fuzzy conditions:* "IF Glucose is High AND BMI is Medium AND Age is
Medium THEN risk = 0.85."
📎 `results/rules.md`

**1.8 What is a neural network?**
*Layers of neurons; each computes a weighted sum plus a bias, then a non-linear
activation. Trained by backpropagation.* Ours: 8 inputs → 8 ReLU neurons → 1
sigmoid output, 81 parameters.

**1.9 What is a neuro-fuzzy system?**
*A fuzzy rule system whose parameters are learned from data like a neural
network's.* Ours is an ANFIS.

**1.10 What is ANFIS?**
*Adaptive Neuro-Fuzzy Inference System (Jang, 1993): a Takagi-Sugeno fuzzy
system written as a 5-layer network: fuzzification, rule firing, normalisation,
consequents, output.*

**1.11 What dataset? What does a row mean?**
*Pima Indians Diabetes, 768 women aged 21+, 8 measurements, outcome diabetic
yes/no.* 35 % are diabetic.

**1.12 What is a black box?**
*A model we can query but whose internal reasoning we cannot interpret.* Our code
only calls the network's `predict_proba`.

---

## Level 2 · Methodology

**2.1 ⭐ Why train the explainer on the network's outputs, not the true labels?**
*Because we want to explain the network; trained on true labels it would be a
different model explaining itself.*
**More detail:** the ablation shows fidelity falls from 94.8 % to 88.2 % when
trained on true labels.
📎 `summary.md` §5

**2.2 ⭐ Mamdani or Sugeno? Why?**
*Zero-order Takagi-Sugeno: each THEN-part is one number.* It is differentiable,
so ANFIS can learn it; it needs no defuzzification; and "risk = 0.85" is easy to read.

**2.3 Where is defuzzification?**
*In a Sugeno system the weighted average $y = \sum \bar w_r p_r$ (layer 5)
replaces it.*

**2.4 Which t-norm for AND? Why?**
*Product. It is differentiable everywhere, so every condition receives a
gradient; min would pass a gradient to only one condition.* OR is not used.

**2.5 ⭐ Explain hybrid learning.**
*Each epoch: (1) with the membership functions fixed, the output is linear in the
rule outputs, so we solve for them exactly by least squares; (2) with those fixed,
we update centres and widths by gradient descent (Adam).*
📎 `anfis.py::_train`, lines 270–282

**2.6 Where do the rules come from?**
*Fuzzy C-Means clusters the patients together with the network's output; each of
the 8 cluster centres is translated into words by choosing the best-matching term
per input; duplicates are merged.*
📎 `rulebase.py::fcm_rules`

**2.7 Why not a grid of all combinations?**
*$3^4 = 81$ rules (6,561 for all 8 inputs), most of which never fire.* The grid
reference reaches 94.2 % with 324 conditions; ours about 94.8 % with 13.

**2.8 How does FCM differ from k-means?**
*Soft memberships that sum to 1 per point, instead of one hard cluster; the
fuzzifier m = 2 controls softness.*

**2.9 How are the initial membership functions placed?**
*Centres at the 10/50/90 % quantiles; widths so that neighbouring terms cross at
0.5.*

**2.10 ⭐ What constraints do you apply and why?**
*Centres may move at most 0.1 of the range; stay ordered Low < Medium < High;
keep at least half their original gap; widths stay between ½× and 2×. Without
them, terms drift by about half the range and swap order in 2 of 5 runs, with no
gain in fidelity.*
📎 `anfis.py::_apply_constraints`; `summary.md` §5

**2.11 Why shoulders?**
*So glucose 200 is fully "High". A bell-shaped High would say 200 is less High
than 155.*

**2.12 How does pruning work?**
*Try deleting every condition, re-fit the rule outputs, and keep the best deletion
if validation fidelity drops by at most 1 point and the probability error grows
by at most 25 %. Repeat, then do the same for whole rules, then fine-tune.*

**2.13 Why the second (error) check?**
*Fidelity only counts flipped decisions; alone, it lets pruning push patients
next to 0.5, where they flip on new data.*
**Careful:** the "92 % instead of 94 %" figure comes from development runs and is
not reproduced by the script.

**2.14 What is oracle sampling? Is it cheating?**
*We add noise to real training patients and ask the network for its answer. Not
cheating: the target is the network, so its answer is correct by definition, and
no test patient is used.* It adds 1.3 points.

**2.15 Why only 4 inputs?**
*Rules with 8 conditions are hard to read; the top 4 carry 83 % of the network's
measured importance; the input-count study found 4 best on average (small
differences).*

**2.16 How did you preprocess the data?**
*Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI mean "not
measured", so we replaced them with the median of the measured training values.
Then a stratified 60/20/20 split. The network standardises inputs; ANFIS scales
to [0, 1].*

**2.17 What metrics do you report?**
*Fidelity (overall, near the boundary, far from it), probability gap, accuracy,
F1, ROC-AUC, and size: rules, conditions, rule length; plus term drift and
ordering, and the weak-match share.*

**2.18 How did you choose the network?**
*5-fold cross-validated ROC-AUC over 16 settings; best: 8 hidden neurons, α = 1.0.*

---

## Level 3 · Implementation

**3.1 ⭐ Walk me through the code that runs when I press Train.**
*`webapp/jobs.py::_run`: load_dataset → make_mlp.fit (streaming epochs) →
permutation_importance → fit_fuzzy_surrogate (oracle pools → ANFIS.fit with FCM
rules → simplify → fine_tune) → evaluate → send the results to the browser.*

**3.2 Where is layer 2 computed? Why logarithms?**
*`anfis.py::_log_firing`. We add log-memberships instead of multiplying, because
a product of many small numbers can underflow to 0.*

**3.3 Show me the gradient code.**
*`anfis.py::_premise_gradients`: $G_{nr} = \frac{2}{N}(y-t)\bar w(p-y)$, summed
over the rules that use each term, times $d/\sigma^2$ for centres or $d^2/\sigma^3$
for widths.*

**3.4 How do you know the gradient is right?**
*`tests/test_anfis.py::test_analytic_gradient_matches_numerical` compares it with
finite differences for every parameter (relative tolerance 1e-4).*

**3.5 Which parts did you write yourselves, and which come from libraries?**
*ANFIS, FCM, membership functions, pruning, the Adam optimiser for ANFIS,
explanations and the lab are our NumPy/JavaScript code; scikit-learn provides the
network, splitting, CV, the decision tree and metrics.*
**Be honest** about any AI or other help you used, according to your college's policy.

**3.6 How is the network made to stream its epochs?**
*`StreamingMLP` overrides a method scikit-learn calls once per epoch and calls our
callback; the training maths is unchanged, and we verified identical predictions.*
Risk: it is a private scikit-learn method.

**3.7 How do I reproduce your numbers?**
*`python scripts/run_experiments.py` (about 5 minutes) regenerates `results/`.
The lab with default settings reproduces the main model exactly.*

**3.8 What is the shape of `antecedents_` and what does −1 mean?**
*One row per rule, one column per input; the value is the term index; −1 means
the rule does not use that input.* Final model: 4 × 4.

**3.9 Where are hyperparameters set?**
*`glassbox/config.py`; the lab overrides them from the settings form.*

**3.10 What happens if a rule never fires?**
*The ridge term pulls its output towards the average risk (0.35) instead of an
arbitrary value; pruning will usually delete it.*

**3.11 Why is the validation pool partly synthetic?**
*154 real patients are few; adding 1,000 network-labelled neighbours makes early
stopping and pruning decisions less sensitive to one or two patients.*

**3.12 What do the tests cover?**
*26 tests: data split and imputation, FCM properties, ANFIS layers, gradients,
constraints, rule text, exact explanation decomposition, pruning, and the lab API.*

**3.13 What did you debug along the way?** *(Name real ones.)*
(1) A plain Gaussian made extreme values less "High" → shoulders.
(2) Terms drifted and lost their meaning → constraints.
(3) Pruning pushed patients next to 0.5 → the error check.
(4) With validation-based early stopping the network's results jumped between
runs (one setting even predicted "non-diabetic" for everyone) → switched to
stopping on the training loss.
(5) A possible Windows encoding problem → results are written as UTF-8.
*(Points 1–3 and 5 are visible in commit messages. For point 4 only the
instability is recorded, in the `make_mlp` docstring; the collapsed run was
observed during development and not saved.)*

---

## Level 4 · Critical analysis

**4.1 ⭐ Your network is only 70 % accurate. Why explain it?**
*The project's question is how to explain a given network, so the method does
not depend on the network being excellent. A faithful explanation would also
reveal a poor model's weaknesses.*
**Be candid:** it is only 5 points above always answering "non-diabetic"
(64.9 %) and matches logistic regression (70.8 %).

**4.2 ⭐ Is 92 % fidelity good? Compared to what?**
*A constant explainer gets 67.5 %; a same-size decision tree 87.7 %; the 81-rule
grid ANFIS 94.2 %. Averaged over five other seeds ours is 94.8 % ± 1.1.*

**4.3 ⭐ Is your explainer significantly better than the tree?**
*Not on its own: on 154 patients, McNemar p = 0.21. But it is higher at all nine
sizes in the sweep, which is consistent evidence, not proof.* Also the tree had
less training data, so the comparison favours us.

**4.4 Where does it fail?**
*Near the network's decision boundary (71 % fidelity for the 35 patients with
network probability 0.35–0.65) and for unusual patients (15 % weak matches).*

**4.5 ⭐ Are your terms really meaningful?**
*Ordered and anchored, yes; but some overlap heavily because widths may double:
BMI 37 is Low 0.52 and High 0.53 at the same time. That is a limitation; tighter
width limits or an overlap constraint would fix it.*

**4.6 How would it behave on noisy or unseen data?**
*Inputs beyond the training range are fully Low or High (shoulders); very unusual
combinations get a weak-match warning; we have no test on a different population.*

**4.7 What is the computational cost?**
*Per epoch $O(N R k)$ + least squares $O(NR^2 + R^3)$; a whole run takes about
5 seconds on a laptop. Grid ANFIS grows as $3^k$, which is why we avoided it.*

**4.8 Why not SHAP or LIME?**
*They explain one prediction at a time with numbers per feature; we wanted global
rules in words. We did not compare against them; that is future work.*

**4.9 Is your test set big enough?**
*154 patients: one patient is 0.65 points. That is why we average five seeds and
report spreads; significance testing and repeated splits would strengthen it.*

**4.10 The rules change between seeds. Which is "the" explanation?**
*Each is a different faithful summary; explanation stability is an open research
question we did not solve.*

**4.11 Isn't permutation importance biased?**
*It can be with correlated inputs (Age and Pregnancies). Glucose's lead (44 %) is
large enough that the top choice is not in doubt.*

**4.12 Why did early stopping pick epoch 21 of 300?**
*Validation error stopped improving early; later epochs kept fitting the training
pool. Fewer epochs would give the same model faster.*

---

## Level 5 · Challenging examiner questions

**5.1 ⭐ Why neuro-fuzzy instead of a plain neural network?**
*The plain network is what we are explaining; it cannot give reasons. A
neuro-fuzzy surrogate gives readable rules while still being learned from data.*

**5.2 Why is fuzzy logic necessary here? Why not a decision tree?**
*Not strictly necessary, but better suited: fuzzy rules change smoothly like the
network, use words with graded membership, and in our sweep were more faithful
than trees at every size (not significant individually).*

**5.3 ⭐ What exactly is learned during training?**
*Network: 81 weights. Explainer: 24 premise parameters (centres and widths), the
rule outputs (4 after pruning), and, through FCM and pruning, which conditions and
rules exist.*

**5.4 ⭐ Which parameters are fixed and which trainable?**
*Trainable: weights, centres, widths, rule outputs. Fixed hyperparameters: network
size and α (chosen by CV), 3 terms, 8 clusters, the constraint limits, the
learning rate, epochs, the pruning tolerance, and the 4 inputs.*

**5.5 How do you know it learned something meaningful?**
*(1) 92 % fidelity vs a 67.5 % constant baseline on unseen patients; (2) the rules
match common risk factors (high glucose, high BMI → high risk) 🔎; (3) the
ablations show each component's effect.* We cannot prove medical meaning without
experts.

**5.6 What if the membership functions or rules are poorly chosen?**
*Fidelity drops and explanations become unreliable. We saw this: 2 clusters give
89 %; unconstrained terms lose their meaning; trained on true labels gives 88 %.
The weak-match flag shows when rules do not cover a patient.*

**5.7 ⭐ How would you show the neuro-fuzzy part adds value?**
*Compare against simpler explainers at the same size (tree: lower at every size),
remove parts (ablations), and, missing so far, a user study showing people
understand the network better with our rules.*

**5.8 ⭐ What is the strongest limitation?**
*One small dataset and a small test set, so the differences are not statistically
established; and interpretability is measured by proxies, not people.*

**5.9 What would you do with another month?**
*A fair tree baseline on the same data; significance tests and repeated splits;
an overlap constraint on terms; a SHAP comparison; a second dataset; a small user
study.* (Chapter 11, §11.3)

**5.10 ⭐ Which parts of the proposal did you implement, modify or omit?**
*Implemented: the network, FCM + ANFIS, the fidelity and interpretability metrics,
the trade-off curve, case studies. Modified: clustering inputs plus the network's
output instead of hidden activations; constraints added. Omitted: practical
usability against the raw black box and clinician validation.* (Chapter 2, §2.5)

**5.11 Isn't a surrogate just another model? How can it explain the network?**
*It explains the network only to the extent that it agrees with it: 92 % of
decisions. For the other 8 % the explanation is of the surrogate, not the network,
so we report where disagreements happen.*

**5.12 Could your explanation be misleading for an individual patient?**
*Yes, if the explainer disagrees with the network or no rule fits; that is why the
lab shows agreement and match strength for every patient.*

**5.13 Is this novel?**
*No. The building blocks are established; our contribution is combining and
evaluating them carefully at course scale.*

**5.14 Who wrote this code?**
Answer truthfully. If you used AI assistance or other help, say so and show that
you understand every part, according to your college's rules.

[Next: Chapter 14 · Speaking guide →](14-speaking-guide.md)
