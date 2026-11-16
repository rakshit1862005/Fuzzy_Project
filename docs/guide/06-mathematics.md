# Chapter 6 · The Mathematics of Our Neuro-Fuzzy Method

[← Chapter 5](05-architecture.md) · [Guide index](README.md) · [Next: Data →](07-data.md)

Each equation below is followed by **symbols**, **intuition**, **a number**,
**what it feeds into** and **where it lives in the code**. Examples marked
*(illustrative)* use made-up numbers. Examples marked *(real)* use the trained
main model (report settings, seed 42). The complete real calculation for one
patient is in [Chapter 12](12-worked-example.md).

---

## 6.0 Notation (used throughout the guide)

| Symbol | Meaning | Our value |
|---|---|---|
| $N$ | number of training examples in the pool | 2,460 (460 real + 2,000 synthetic) |
| $k$ | number of inputs given to ANFIS | 4 (Glucose, BMI, Age, Pregnancies) |
| $j$ | index of an input, $1..k$ | |
| $m$ | number of terms per input | 3 |
| $t$ | index of a term: 0 = Low, 1 = Medium, 2 = High | |
| $R$ | number of rules | 8 before pruning, 4 after |
| $x_j$ | input $j$ in original units (mg/dL, kg/m², years, count) | |
| $\tilde x_j$ | input $j$ scaled to $[0,1]$ | |
| $c_{jt}, \sigma_{jt}$ | centre and width of term $t$ of input $j$ (scaled units) | premise parameters |
| $\mu_{jt}$ | membership degree of $x_j$ in term $t$ | |
| $A_{rj}$ | the term rule $r$ uses for input $j$, or "don't care" (−1) | antecedent matrix |
| $w_r$, $\bar w_r$ | raw and normalised firing strength of rule $r$ | |
| $p_r$ | output (THEN-part) of rule $r$ | consequent parameters |
| $y$ | fuzzy output (risk) | |
| $t_n$ | target for example $n$: the network's probability $f_{NN}(x^{(n)})$ | |

---

## 6.1 Scaling inputs to [0, 1]

$$\tilde x_j = \frac{x_j - \text{lo}_j}{\text{hi}_j - \text{lo}_j}$$

- **Symbols:** $\text{lo}_j$, $\text{hi}_j$ are the minimum and maximum of input $j$ in the training pool.
- **Intuition:** puts every input on the same 0–1 scale, so one learning rate and
  one drift limit (0.1) mean the same for glucose and for age.
- **Real:** Glucose lo = 56, hi = 197. Glucose 184 → (184 − 56)/141 = **0.908**. ✅
- **Feeds into:** every membership calculation.
- **Code:** `anfis.py`, `ANFIS._scale` (line 106); lo and hi are set in `fit` (lines 226–228).

## 6.2 Initial terms

Centres at the 10 %, 50 % and 90 % quantiles of the scaled training data; one
shared width per input, chosen so that neighbouring terms cross at membership 0.5:

$$c_{j0}, c_{j1}, c_{j2} = Q_{0.1}, Q_{0.5}, Q_{0.9} \qquad \sigma_j = \frac{\overline{\Delta c}}{2\sqrt{2\ln 2}} \approx \frac{\overline{\Delta c}}{2.355}$$

- **Symbols:** $Q_q$ is the $q$-quantile; $\overline{\Delta c}$ is the average gap between neighbouring centres.
- **Why $2\sqrt{2\ln2}$:** a Gaussian falls to 0.5 at distance $\sqrt{2\ln 2}\,\sigma$
  from its centre. Halfway between two centres (distance $\overline{\Delta c}/2$),
  both terms then have membership exactly 0.5. So every value belongs at least
  "half" to some term.
- **Why quantiles:** each term covers a similar number of patients even for
  skewed inputs.
- **Real (Glucose):** initial centres 0.228, 0.444, 0.762 (88, 119, 163 mg/dL);
  average gap 0.267; $\sigma = 0.267/2.355 = 0.113$. ✅
- **Code:** `membership.py::initial_partition` (lines 47–63).

## 6.3 Membership functions (layer 1), with shoulders

$$\mu_{jt}(\tilde x_j) = \exp\!\left(-\frac{d_{jt}^{\,2}}{2\sigma_{jt}^2}\right), \qquad
d_{jt} = \begin{cases}
\max(\tilde x_j - c_{jt},\,0) & t = \text{Low (shoulder: flat below its centre)}\\
\tilde x_j - c_{jt} & t = \text{Medium}\\
\min(\tilde x_j - c_{jt},\,0) & t = \text{High (shoulder: flat above its centre)}
\end{cases}$$

- **Intuition:** Medium is a bell. Low and High are half-bells that stay at 1
  beyond their centres, so very high glucose is fully "High".
- **Real:** glucose 0.908, Medium centre 0.544, σ = 0.112:
  $d = 0.364$, $\mu = e^{-0.5 (0.364/0.112)^2} = e^{-5.25} = $ **0.005**.
  High centre 0.703: 0.908 is above it, so $d = 0$ and $\mu = $ **1.00**. ✅
- **Feeds into:** rule firing.
- **Code:** `membership.py::term_offsets` (computes $d$ with shoulders) and
  `log_memberships` (lines 21–38).

## 6.4 Rules and antecedents

A rule base is an integer matrix $A$ of shape $R \times k$: $A_{rj} = t$ means
"input $j$ is term $t$"; $A_{rj} = -1$ means rule $r$ ignores input $j$.

*(Real)* the final rule base (inputs: Glucose, BMI, Age, Pregnancies):

| Rule | $A_r$ | Reads as |
|---|---|---|
| R1 | [0, 0, 0, 0] | IF Glucose Low AND BMI Low AND Age Low AND Pregnancies Low |
| R2 | [1, −1, −1, 1] | IF Glucose Medium AND Pregnancies Medium |
| R3 | [1, 2, 1, −1] | IF Glucose Medium AND BMI High AND Age Medium |
| R4 | [2, 1, 1, −1] | IF Glucose High AND BMI Medium AND Age Medium |

✅ (computed from the trained model; matches `results/rules.md`)

## 6.5 Fuzzy C-Means: where the rules come from

FCM clusters the training pool in the joint space (4 scaled inputs + the network's
output), so a cluster is a group of patients who *look alike and get a similar
decision*. It minimises

$$J_m = \sum_{n=1}^{N}\sum_{i=1}^{c} u_{ni}^{\,m}\,\lVert z_n - v_i\rVert^2 \quad\text{subject to}\quad \sum_{i} u_{ni} = 1$$

by alternating two updates until the memberships change by less than $10^{-6}$:

$$v_i = \frac{\sum_n u_{ni}^m z_n}{\sum_n u_{ni}^m}, \qquad
u_{ni} = \left[\sum_{l=1}^{c}\left(\frac{\lVert z_n - v_i\rVert}{\lVert z_n - v_l\rVert}\right)^{\frac{2}{m-1}}\right]^{-1}$$

- **Symbols:** $z_n$ = (scaled inputs, network output) of example $n$; $v_i$ = centre of cluster $i$; $c$ = number of clusters (8); $m$ = fuzzifier (2); $u_{ni}$ = membership of example $n$ in cluster $i$.
- **Intuition:** centres are weighted averages of the points, with weights
  $u^m$. Memberships are high for the nearest centre and low for far ones.
- **Example (illustrative, 1-D):** points 1, 2, 9, 10 and centres 1.5, 9.5.
  Point 2 is 0.5 from centre 1 and 7.5 from centre 2:
  $u = 1/\big(1 + (0.5/7.5)^2\big) = $ **0.9956** for cluster 1, 0.0044 for cluster 2.
  Re-computing the centres gives 1.4996 and 9.5004, so the algorithm has converged.
- **From cluster to rule:** for each centre and each input, choose the term with
  the highest membership at the centre's coordinate. Clusters that give the same
  sentence are merged.
- **Real:** 8 clusters gave 8 distinct rules; FCM ran 252 iterations; the
  objective fell from 70.4 to 54.8. ✅
- **Code:** `fcm.py::fuzzy_c_means` (lines 34–50), `_memberships` (27–31);
  `rulebase.py::fcm_rules` (lines 30–56).

## 6.6 Rule firing strength (layer 2): AND = product

$$w_r = \prod_{j:\,A_{rj}\neq -1} \mu_{j,A_{rj}}(\tilde x_j) \qquad\Longleftrightarrow\qquad \log w_r = \sum_{j:\,A_{rj}\neq -1} \log \mu_{j,A_{rj}}$$

- **Intuition:** a rule fires strongly only if *all* its conditions hold. One
  weak condition pulls the product down.
- **Why logs:** multiplying many small numbers can underflow to 0 in a computer;
  adding their logarithms gives the same result safely.
- **Real:** R4 for patient #152: $1.00 \times 1.00 \times 0.93 = $ **0.93**. ✅
- **Code:** `anfis.py::_log_firing` (lines 123–135).

## 6.7 Normalisation (layer 3)

$$\bar w_r = \frac{w_r}{\sum_{q=1}^{R} w_q}$$

- **Intuition:** each rule's *share* of the decision; the shares sum to 1.
- **Code:** `anfis.py::_normalize` (lines 137–140) computes this from the logs as
  $\exp(\log w_r - \max_q \log w_q)$ divided by the sum. Subtracting the maximum
  avoids overflow and does not change the ratio.
- **Real:** for patient #152, $\sum w = 0.948$ and $\bar w_4 = 0.933/0.948 = $ **0.984**. ✅
- **Caution:** normalisation hides how *well* the best rule fits. If all rules
  fire at 0.001, one still gets a large share. That is why we also report the
  **match strength** $\max_r w_r$ (§6.16).

## 6.8 Consequents and output (layers 4–5)

$$y = \sum_{r=1}^{R} \bar w_r\, p_r, \qquad \text{decision} = \begin{cases}\text{Diabetic} & y \ge 0.5\\ \text{Non-diabetic} & y < 0.5\end{cases}$$

- **Intuition:** a weighted average of what each rule says, weighted by how much
  each rule applies. This is Takagi-Sugeno inference; there is no separate
  defuzzification step.
- **Real:** patient #152: $0.008(0.033) + 0.005(0.352) + 0.003(0.853) + 0.984(0.852) = $ **0.843**. ✅
- **Code:** `anfis.py::_raw_output` (line 293), `predict_proba` (clips to [0, 1],
  line 300), `predict` (threshold 0.5, line 304).

## 6.9 What ANFIS minimises

$$L = \frac{1}{N}\sum_{n=1}^{N}\big(y_n - t_n\big)^2, \qquad t_n = f_{NN}(x^{(n)})$$

- **Intuition:** the squared gap between the fuzzy risk and the **network's**
  probability, averaged over the pool. The true diagnosis does not appear.
- **Code:** `anfis.py::_train`, line 274 (training loss) and line 276 (validation loss).

## 6.10 Hybrid learning step 1: consequents by least squares

With the premises fixed, $y$ is **linear** in $p$: $y = \bar W p$, where $\bar W$
is the $N \times R$ matrix of normalised firing strengths. So the best $p$ has a
closed form. We add a small ridge penalty pulling each $p_r$ towards the average
target $\bar t$:

$$p = \big(\bar W^\top \bar W + \lambda I\big)^{-1}\big(\bar W^\top t + \lambda\,\bar t\,\mathbf 1\big), \qquad \text{then clip each } p_r \text{ to } [0,1]$$

- **Symbols:** $\lambda = 1$ (`ridge`); $\bar t$ = mean target (0.352); $\mathbf 1$ = vector of ones.
- **Intuition:** "solve for the rule outputs exactly". The ridge term acts like one
  imaginary average patient per rule, so a rule that hardly ever fires gets a
  neutral output instead of an arbitrary one. With 2,460 examples its effect is
  small.
- **Example (illustrative):** 2 rules, 3 patients with targets (0.30, 0.05, 0.90)
  and normalised firing strengths (0.848, 0.152), (0.996, 0.004), (0.0005, 0.9995).
  Without ridge, $p = (0.107, 0.911)$. With $\lambda = 1$ and only 3 patients, the
  ridge pulls strongly: $p = (0.234, 0.658)$, towards the mean 0.417.
- **Caution:** clipping makes this a *projected* least-squares solution. If a
  value needed clipping, it is no longer the exact optimum. 🔎
- **Code:** `anfis.py::_solve_consequents` (lines 150–171).

## 6.11 Hybrid learning step 2: premises by gradient descent

We need $\partial L/\partial c_{jt}$ and $\partial L/\partial \sigma_{jt}$, holding $p$ fixed.

**Step 1: the output with respect to one rule's log-firing.** Using the
quotient rule on $\bar w_r = w_r/\sum_q w_q$:

$$\frac{\partial y}{\partial \theta} = \sum_r \bar w_r\,(p_r - y)\,\frac{\partial \log w_r}{\partial \theta}$$

*Intuition:* making rule $r$ fire more strongly pulls $y$ towards $p_r$. Whether
that helps depends on the sign of $p_r - y$.

**Step 2: the log-membership with respect to its parameters.** From §6.3, $\log\mu = -d^2/(2\sigma^2)$, so

$$\frac{\partial \log\mu}{\partial c} = \frac{d}{\sigma^2}, \qquad \frac{\partial \log\mu}{\partial \sigma} = \frac{d^2}{\sigma^3}$$

On the flat side of a shoulder $d = 0$, so the gradient is 0.

**Step 3: combine with $\partial L/\partial y_n = 2(y_n - t_n)/N$:**

$$\frac{\partial L}{\partial c_{jt}} = \sum_{n}\;\sum_{r:\,A_{rj}=t} G_{nr}\,\frac{d_{n,jt}}{\sigma_{jt}^2}, \qquad
\frac{\partial L}{\partial \sigma_{jt}} = \sum_{n}\;\sum_{r:\,A_{rj}=t} G_{nr}\,\frac{d_{n,jt}^2}{\sigma_{jt}^3}, \qquad
G_{nr} = \frac{2}{N}(y_n - t_n)\,\bar w_{nr}\,(p_r - y_n)$$

The sum over $r$ includes only rules that use term $t$ of input $j$. Because
terms are shared, one term collects gradient from every rule that mentions it.

**Example (illustrative):** same 3 patients, rule 2 = "IF $x_1$ is High",
High centre 0.7, σ = 0.15, $p = (0.107, 0.911)$. The two terms that matter
(the third patient lies on the flat part of the shoulder, $d = 0$):

$$\frac{\partial L}{\partial c} = \frac{(-0.0049)(-0.3) + (0.0001)(-0.5)}{0.15^2} \approx 0.062$$

A finite-difference check gives the same value (0.06238 vs 0.06238).

*Interpretation:* the gradient is positive, so gradient descent *lowers* the High
centre. Patient A (x₁ = 0.4) then becomes more "High", gives more weight to rule 2
(output 0.91), and its prediction 0.229 rises towards its target 0.30. The maths
does what intuition says it should.

- **Verified in tests:** `tests/test_anfis.py::test_analytic_gradient_matches_numerical`
  checks every centre and width gradient against finite differences (relative
  tolerance $10^{-4}$). ✅
- **Code:** `anfis.py::_premise_gradients` (lines 173–198).

## 6.12 The optimiser: Adam

For each parameter $\theta$ with gradient $g$ at step $s$:

$$m \leftarrow \beta_1 m + (1-\beta_1) g, \quad v \leftarrow \beta_2 v + (1-\beta_2) g^2, \quad
\hat m = \frac{m}{1-\beta_1^s}, \quad \hat v = \frac{v}{1-\beta_2^s}, \quad
\theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat v} + \epsilon}$$

- **Values:** $\eta = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. ✅
- **Intuition:** $m$ smooths the gradient direction (momentum); $\sqrt{v}$ scales
  each step by the typical gradient size. On the first step $\hat m/\sqrt{\hat v} = \pm 1$,
  so every centre moves by about 0.01. Movement is roughly bounded by 0.01 per epoch.
- **Code:** `anfis.py::Adam` (lines 46–64).

## 6.13 Constraints that keep the words meaningful

After every update, `_apply_constraints` (lines 200–214) projects the parameters back into an allowed region:

| # | Constraint | Formula | Why |
|---|---|---|---|
| 1 | limited drift, inside the data range | $c_{jt} \in [\max(c^0_{jt} - 0.1,\,0),\ \min(c^0_{jt} + 0.1,\,1)]$ | "Low" must stay low. |
| 2 | ordering | sort centres so Low < Medium < High (widths move with their centres) | Labels stay correct. |
| 3 | minimum gap | $c_{j,t} \ge c_{j,t-1} + 0.5\,(c^0_{j,t} - c^0_{j,t-1})$ | Terms may not merge. |
| 4 | width range | $\sigma_{jt} \in [0.5\,\sigma^0_{jt},\ 2\,\sigma^0_{jt}]$ | No spikes; no term swallowing its neighbours. |

($c^0$, $\sigma^0$ are the initial values.)

- **Real:** several Glucose centres moved by exactly the limit +0.1 (Low 0.228 → 0.328,
  Medium 0.444 → 0.544), and the Glucose Low width grew from 0.113 to 0.206, almost
  the 2× limit. ✅ Constraint 4 allows this much widening, which is why some terms
  overlap heavily ([Chapter 11](11-results-critique.md)).
- **Measured effect:** without constraints, fidelity does not improve
  (94.2 % vs 94.8 %), while terms drift by 0.50 on average and lose their order in
  2 of 5 runs. ✅ (`results/summary.md`, §5)

## 6.14 Early stopping

Every epoch we compute the validation MSE (against the network, on 154 real +
1,000 synthetic validation patients). At the end, the parameters from the
**epoch with the lowest validation MSE** are restored (`_train`, lines 275–287).

- **Real:** for the main model's first training phase, the best epoch was **21 of 300**. ✅
  Validation error stopped improving very early; later epochs kept fitting the
  training pool without helping on new data.

## 6.15 Pruning (simplification)

Let $F_0$ = validation fidelity and $E_0$ = training-pool MSE of the trained model.
A candidate edit (delete one condition, or one whole rule) is **acceptable** if

$$F \ge F_0 - \tau \quad\text{and}\quad E \le (1 + \gamma)\,E_0, \qquad \tau = 0.01,\ \gamma = 0.25$$

Algorithm (greedy, best-first):
1. For every possible edit, re-fit the consequents by least squares (§6.10) and
   compute $F$ and $E$.
2. Among acceptable edits, apply the one with the highest $F$ (ties: lower $E$).
3. Merge rules that have become identical. Repeat until no edit is acceptable.
4. Do the same for whole rules. Then fine-tune for 100 epochs (§6.10–6.14 again).

- **Why the second check:** fidelity only counts flipped decisions. On its own it
  lets pruning push many patients close to 0.5, where they flip on new data.
  Bounding the probability error prevents that. (The docs report that test
  fidelity averaged about 92 % instead of 94 % without this check in development
  runs; ❓ that comparison is **not** reproduced by `run_experiments.py`, see
  [Chapter 15](15-revision-sheets.md).)
- **Real:** the main model went from 8 rules / 32 conditions to 4 rules / 12
  conditions in 14 accepted edits; validation fidelity stayed between 95.4 % and
  96.3 % throughout. ✅ (the simplification log in `results/rules.md`)
- **Code:** `simplify.py::simplify`, `_greedy`, `_condition_removals`, `_rule_removals`.

## 6.16 Explanation and match strength

$$y = \sum_r \underbrace{\bar w_r p_r}_{\text{contribution of rule } r}, \qquad \text{match strength} = \max_r w_r \quad(\text{weak if } < 0.2)$$

- Contributions add up **exactly** to the output; nothing is hidden.
- Match strength uses the *raw* firing strength, so it shows when no rule really
  fits a patient. 15 % of test patients are weak matches. ✅
- **Code:** `explain.py::explain_instance`; `WEAK_MATCH = 0.2`.

## 6.17 First-order variant (ablation only)

$p_r$ is replaced by $f_r(\tilde x) = b_r + \sum_j a_{rj}\tilde x_j$. Least squares
then has $R(k+1)$ unknowns. Implemented (`order=1`) only to compare: 94.3 % fidelity
vs 94.8 %, with 18 numbers after THEN instead of 5. ✅

---

## 6.18 The black box's mathematics (for completeness)

Our network, for a standardised input $\hat x = (x - \text{mean})/\text{std}$:

$$h = \text{ReLU}(W_1^\top \hat x + b_1) \in \mathbb{R}^8, \qquad \hat y = \sigma(w_2^\top h + b_2), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

Loss on a mini-batch $B$ of 200 patients:

$$L = -\frac{1}{|B|}\sum_{n\in B}\big[y_n\log\hat y_n + (1-y_n)\log(1-\hat y_n)\big] + \frac{\alpha}{2|B|}\big(\lVert W_1\rVert^2 + \lVert w_2\rVert^2\big), \quad \alpha = 1$$

Weights are updated by backpropagation and Adam (learning rate 0.001). Training
stops when the training loss improves by less than $10^{-4}$ for more than 10
consecutive epochs (321 epochs here). ✅ (scikit-learn `MLPClassifier`, checked in its source)

## 6.19 Permutation importance

$$I_j = \frac{1}{30}\sum_{r=1}^{30}\ \frac{1}{|V|}\sum_{n\in V}\big|\,f_{NN}(x^{(n)}_{\text{col } j \text{ shuffled},\,r}) - f_{NN}(x^{(n)})\,\big|$$

Shuffle column $j$ of the validation set $V$, measure the average absolute change
in the network's output, repeat 30 times and average. Glucose: 0.231 (44 % of the
total). ✅ (`feature_selection.py`, lines 14–28)

## 6.20 Oracle sampling

$$x^{\text{syn}} = \text{clip}\big(x^{(n)} + 0.15\,\epsilon \odot s,\ \text{min},\ \text{max}\big), \qquad \epsilon \sim \mathcal N(0, I),\quad t^{\text{syn}} = f_{NN}(x^{\text{syn}})$$

$x^{(n)}$ is a randomly chosen real training patient, $s$ is the vector of feature
standard deviations, and $\odot$ multiplies element by element. ✅ (`oracle.py`, lines 14–29)

[Next: Chapter 7 · Data →](07-data.md)
