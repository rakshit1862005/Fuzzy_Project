# Chapter 14 · Presentation and Viva Speaking Guide

[← Chapter 13](13-viva-questions.md) · [Guide index](README.md) · [Next: Revision sheets →](15-revision-sheets.md)

Say these out loud several times. Every number below is in `results/`.

---

## 14.1 The 30-second introduction

> "Our project is called Glass-Box Insight. A neural network can predict
> diabetes, but it can't tell you why. We built a neuro-fuzzy system, an ANFIS,
> that learns to imitate the network with four readable IF-THEN rules. On
> patients it has never seen, it makes the same decision as the network 92 % of
> the time, and for any patient we can show exactly which rule produced the answer."

## 14.2 The two-minute explanation

> "**The problem.** Neural networks are accurate but opaque. In medicine an
> unexplained decision is hard to trust or check. Fuzzy rule systems are
> readable, 'IF glucose is high THEN risk is high', but they usually need an
> expert to write them.
>
> **Our idea.** Treat the trained network as a black box we can question. Train a
> fuzzy system to give the *network's* answers, not the true diagnoses, so its
> rules describe the network.
>
> **How.** We use the Pima diabetes data: 768 patients, 8 measurements. After
> cleaning missing values we train a small network, 8 inputs, 8 hidden neurons,
> 1 output. We measure which inputs it relies on most, which are glucose, BMI,
> age and pregnancies. Fuzzy C-Means clustering proposes rules in words. ANFIS
> tunes them with hybrid learning: least squares for the rule outputs, gradient
> descent for the Low/Medium/High curves, with constraints so the words keep their
> meaning. Then pruning deletes every condition that isn't needed.
>
> **Results.** Four rules, twelve conditions, 92 % agreement with the network on
> unseen patients, compared with 88 % for a decision tree of the same size. For
> scale: a rule base that always gave one fixed answer would already get 67 %.
>
> **Limits.** One small dataset, a network that's only as good as logistic
> regression, some overlapping terms, and no study with doctors yet."

## 14.3 The five-minute technical explanation (with transitions)

1. **Motivation (45 s).** Black box versus readable rules; the goal is
   *fidelity*, not accuracy. → *"So the first thing we need is a black box to explain."*
2. **Data and network (45 s).** Pima data; zeros in five columns mean "missing" →
   training-median imputation; stratified 60/20/20 split; MLP 8→8→1 chosen by
   5-fold CV; 70.1 % accuracy, about the same as logistic regression.
   → *"We never look inside it again; we only ask it questions."*
3. **Choosing inputs (20 s).** Permutation importance on the network's output:
   glucose 44 %; the top four go to the fuzzy system.
   → *"Now we need rules over these four inputs."*
4. **Rules from FCM (40 s).** A grid would need 81 rules; Fuzzy C-Means clusters
   patients together with the network's output; each of 8 centres becomes a rule
   in words over shared Low/Medium/High terms.
   → *"These rules are rough, so we tune them."*
5. **ANFIS and hybrid learning (70 s).** Show the five layers. Least squares for
   the rule outputs because the output is linear in them; gradient descent with
   Adam for centres and widths; the gradient checked numerically; four
   constraints; keep the best validation epoch. Mention the synthetic patients
   labelled by the network.
   → *"Eight rules with four conditions each are still long, so we prune."*
6. **Pruning (30 s).** Delete the condition that hurts least while fidelity stays
   within 1 point and the error grows by at most 25 %; then fine-tune. 32 → 12
   conditions.
   → *"Here is what that gives us."*
7. **Results and limits (50 s).** 92.2 % fidelity; 94.8 % ± 1.1 over five other
   seeds; above trees at every size, though not significant on its own; the
   ablations; the worked example of patient #152; then the limitations.

## 14.4 Explaining the architecture diagram

Use the diagram in [Chapter 1](01-overview.md#16-the-complete-system-at-a-glance)
and point left to right:

> "Data comes in here and is cleaned. This box is the black box: the only thing
> that crosses from it to the right is its *predicted probability*: arrows out of
> the network carry answers, never weights. Those answers do three jobs: rank the
> inputs, label synthetic patients, and serve as the targets for the fuzzy system.
> FCM proposes rules, ANFIS tunes them, pruning shortens them, and at the end we
> have rules, per-patient explanations and an evaluation on patients that no step
> ever saw."

## 14.5 The algorithm step by step (for "explain ANFIS")

1. Scale each input to 0–1.
2. **Layer 1:** compute each term's membership (bell for Medium, shoulders for
   Low and High).
3. **Layer 2:** multiply each rule's memberships: its firing strength.
4. **Layer 3:** divide by the total: each rule's share.
5. **Layers 4–5:** output = Σ share × rule output.
6. **Training, each epoch:** least squares for the rule outputs; gradient of the
   squared error for centres and widths; one Adam step; apply the constraints;
   remember the best validation epoch.

## 14.6 Research motivation, in one breath

> "This sits in explainable AI and neuro-fuzzy modelling. Rule extraction from
> networks is a 30-year-old field, and the interpretability-versus-accuracy
> trade-off is a central theme in fuzzy modelling. We're not claiming anything
> new. We implemented and evaluated a careful combination at course scale."

## 14.7 Presenting results and limitations defensibly

- Lead with **fidelity**, then give the baselines (constant 67.5 %, tree 87.7 %),
  then the seed average.
- Say "consistently higher than trees in our sweep" and **not** "significantly
  better".
- Volunteer the limitations before you are asked: dataset size, the network ≈
  logistic regression, overlapping terms, proxies for interpretability, no user study.
- For every limitation, have the fix ready (Chapter 11, §11.3).

## 14.8 Common mistakes, and how to avoid them

| Mistake | Say instead |
|---|---|
| "Membership 0.7 means a 70 % chance." | "Degree of fit to the word, not a probability." |
| "Our fuzzy model is 92 % accurate." | "92 % **fidelity**: agreement with the network. Accuracy is 70 %." |
| "Fuzzy logic handles uncertainty." | "It models vague category boundaries; we don't model measurement uncertainty." |
| "ANFIS is a neural network." | "A fuzzy system *written as* a network so it can learn." |
| "We used defuzzification (centroid)." | "Sugeno: the weighted average replaces defuzzification." |
| "The rules are medically validated." | "They match common risk factors; no clinician has reviewed them." |
| "Our method is better than decision trees." | "Higher fidelity at every size we tried on this dataset; not significant per comparison." |
| "The explanation shows how the network reasons." | "It shows a faithful approximation that agrees on 92 % of decisions." |
| "We trained on the test set to check …" | Never: the test set is used only for the final numbers. |
| Reading equations without saying what they do | Say what the equation achieves *first*, then show it. |

[Next: Chapter 15 · Revision sheets →](15-revision-sheets.md)
