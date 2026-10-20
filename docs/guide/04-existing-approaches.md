# Chapter 4 · Existing Approaches and Why Ours Was Chosen

[← Chapter 3](03-background.md) · [Guide index](README.md) · [Next: Architecture →](05-architecture.md)

> **Evidence note.** No project-specific research paper was supplied; the project
> is not a reproduction of one paper. This chapter therefore relies on standard,
> well-known references, listed in §4.5 and in
> [`docs/PROBLEM_STATEMENT.md`](../PROBLEM_STATEMENT.md). ❓ If your guide or
> department expects the project to follow a particular paper, find out which one
> before the viva.

---

## 4.1 The question every approach is answering

*"How can a person understand why a trained model made a decision?"*
There are two big families of answers:

1. **Use a model that is readable from the start** (inherently interpretable).
2. **Keep the accurate black box and explain it afterwards** (post-hoc
   explanation). Post-hoc methods are either **local** (explain one prediction)
   or **global** (describe the whole model).

Our project is a **global, post-hoc** method: a **surrogate** that imitates the
whole network, and that also yields exact local explanations for its own
predictions.

## 4.2 The approaches

### A. Inherently interpretable models (logistic regression, small decision trees)

- **Idea:** skip the network and use a readable model directly.
- **How:** logistic regression gives one weight per feature; a decision tree
  gives IF-THEN splits.
- **Strengths:** simple, fast, honest (the explanation *is* the model).
- **Weaknesses:** may be less accurate on complicated data; not applicable if a
  network is *already deployed* and must be explained.
- **In our project:** logistic regression is used as a **reference** for accuracy:
  70.8 % vs the network's 70.1 %. ✅ This is a real point an examiner may raise:
  on this dataset a simple model does as well as the network. Our answer: the
  project's question is *how to explain a given network*, and the method does not
  depend on the network being better.

### B. Local post-hoc explanations: LIME, SHAP, Anchors

- **Idea:** explain one prediction at a time.
  - *LIME* (Ribeiro et al., 2016) fits a small linear model around one input.
  - *SHAP* (Lundberg & Lee, 2017) splits a prediction into additive feature
    contributions based on Shapley values from game theory.
  - *Anchors* (Ribeiro et al., 2018) finds an IF-condition that "anchors" a
    prediction locally.
- **Strengths:** model-agnostic, widely used, per-patient detail.
- **Weaknesses:** they explain one point, not the model's overall logic;
  explanations can vary between nearby inputs; SHAP gives numbers per feature,
  not rules in words.
- **In our project:** ❌ **not implemented and not compared.** This is a known gap
  ([Chapter 11](11-results-critique.md)).

### C. Global surrogate models (decision tree surrogate, TREPAN)

- **Idea:** train a readable model on the *black box's predictions*.
- **TREPAN** (Craven & Shavlik, 1996) builds a decision tree from a network and
  queries the network on extra generated inputs. Our *oracle sampling* borrows
  that idea.
- **Strengths:** a whole-model view; any black box works.
- **Weaknesses:** the surrogate is only an approximation; hard axis-aligned splits
  can need many leaves to follow a smooth decision surface.
- **In our project:** a **decision-tree surrogate** of the same size is the main
  baseline ✅ (`glassbox/surrogates.py`). Our method is itself a global surrogate,
  with fuzzy rules instead of tree splits.

### D. Rule extraction from neural networks

Andrews, Diederich & Tickle (1995) classify methods as:
- **decompositional:** read rules from the network's internal weights and neurons;
- **pedagogical:** treat the network as a black box and learn rules from its
  input–output behaviour.

**Our method is pedagogical** ✅: nothing reads the network's weights. Only
`blackbox.predict_proba` is called.

### E. Hand-built fuzzy systems (Mamdani)

- **Idea:** experts write the rules and membership functions.
- **Strengths:** fully readable; encodes expert knowledge.
- **Weaknesses:** needs an expert; does not learn; does not imitate a specific
  network.
- **In our project:** ❌ not used. We need rules *learned from the network's
  behaviour*.

### F. Data-driven fuzzy systems

| Method | Idea | Weakness | In our project |
|---|---|---|---|
| **Wang–Mendel** (1992) | one rule per training example, keep the most frequent | many rules, no tuning | not used |
| **Grid-partition ANFIS** (Jang, 1993) | every combination of terms is a rule; tune by hybrid learning | $m^k$ rules: 81 for 4 inputs, 6,561 for 8 | ✅ implemented as a **reference** (`rule_init="grid"`): 81 rules, 324 conditions, 94.2 % fidelity |
| **Clustering-based** (FCM; subtractive clustering, Chiu, 1994) | each cluster becomes a rule | clusters can give long rules; per-rule membership functions lose shared meaning | ✅ **FCM used**, but snapped to *shared* Low/Medium/High terms |
| Unconstrained gradient tuning | tune all parameters freely | terms drift and swap meaning | ✅ tested as an **ablation**: drift ≈ 0.50 of the input range, order broken in 2 of 5 runs |

### G. Our approach: a pedagogical, constrained, pruned ANFIS surrogate

FCM proposes rules over shared terms → ANFIS hybrid learning on the network's
probabilities (plus synthetic oracle samples) → constraints keep the terms
meaningful → greedy pruning with two acceptance checks → fine-tuning.

## 4.3 Comparison table

| | Accuracy vs truth | Faithful to the given network? | Global view | Rules in words | Learns from data | Size control | Implemented here |
|---|---|---|---|---|---|---|---|
| Logistic regression | ✅ 70.8 % | n/a (separate model) | ✅ | weights, not rules | ✅ | – | reference only |
| Small decision tree on truth | moderate | n/a | ✅ | ✅ hard splits | ✅ | ✅ | ❌ |
| LIME / SHAP | n/a | locally | ❌ | ❌ numbers | n/a | – | ❌ |
| Decision-tree surrogate | n/a | ✅ 87.7 % at 4 leaves | ✅ | ✅ hard splits | ✅ | ✅ | ✅ baseline |
| Hand-built Mamdani | depends | ❌ | ✅ | ✅ | ❌ | manual | ❌ |
| Grid ANFIS surrogate | n/a | ✅ 94.2 % | ✅ | 81 long rules | ✅ | ❌ | ✅ reference |
| **Ours** | 70.1 % (≈ network) | ✅ **92.2 %** at 4 rules (94.8 % ± 1.1 over 5 seeds) | ✅ | ✅ graded rules | ✅ | ✅ pruning | ✅ |

All fidelity numbers are on the same 154 test patients. ✅ (`results/summary.md`)

## 4.4 Why this approach was chosen

| Requirement (from the problem statement) | Why our choice fits |
|---|---|
| Explain a *given* network | Pedagogical surrogate: works with any black box, no internal access needed. |
| Human-readable rules | Fuzzy IF-THEN rules over named terms with units. |
| Learn rules automatically | FCM proposes rules; ANFIS tunes them. |
| Small rule base | FCM avoids the 81-rule grid; pruning removes what is not needed. |
| Words must keep their meaning | Shared terms plus constraints (drift, ordering, gap, width). |
| Course fit: neural networks + fuzzy logic + ANFIS | Uses all three, implemented from scratch so each step can be explained. |
| Smooth, graded explanations | Fuzzy memberships change smoothly; trees jump at thresholds. |

**Trade-offs we accepted:**
- *Zero-order consequents* (one number per rule) are more readable than linear
  formulas. We measured the alternative: first-order rules were not more
  faithful (94.3 % vs 94.8 %) and need 18 numbers after THEN instead of 5. ✅
- *Only 4 of 8 inputs* keeps rules short; it could miss effects from the other
  four (they carry 17 % of the network's measured importance). ✅
- *Constraints* limit how far learning can move the terms; the ablation shows
  this costs no fidelity (94.8 % vs 94.2 % without). ✅

## 4.5 Research gap and novelty: what we can honestly claim

- **What earlier work established:** rule extraction from networks (Andrews et
  al., 1995; Craven & Shavlik, 1996), ANFIS and hybrid learning (Jang, 1993),
  clustering-based fuzzy modelling (Bezdek, 1981; Chiu, 1994), and the
  interpretability–accuracy trade-off in fuzzy modelling and how to measure it
  (Casillas et al., 2003; Gacto et al., 2011).
- **What our project does:** a course-scale implementation and evaluation that
  combines these: a pedagogical fuzzy surrogate with meaning-preserving
  constraints and two-check pruning, compared with a decision-tree surrogate and
  studied by ablation.
- **Classification:** *implements and evaluates established techniques*, with
  small engineering additions (the probability-error check during pruning; the
  weak-match flag). ❓ We have not searched the literature to establish whether
  these additions are new. **Do not claim novelty.**
- **Is a research gap addressed?** 🔎 The project touches a real open problem
  (faithful *and* readable explanations), but on one small dataset without user
  evaluation. It is a preliminary study, not a solution.

### References

- Andrews, R., Diederich, J., & Tickle, A. B. (1995). Survey and critique of techniques for extracting rules from trained artificial neural networks. *Knowledge-Based Systems*, 8(6), 373–389.
- Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms*. Plenum.
- Casillas, J., Cordón, O., Herrera, F., & Magdalena, L. (Eds.) (2003). *Interpretability Issues in Fuzzy Modeling*. Springer.
- Chiu, S. L. (1994). Fuzzy model identification based on cluster estimation. *Journal of Intelligent & Fuzzy Systems*, 2(3), 267–278.
- Craven, M., & Shavlik, J. (1996). Extracting tree-structured representations of trained networks. *Advances in Neural Information Processing Systems 8*.
- Gacto, M. J., Alcalá, R., & Herrera, F. (2011). Interpretability of linguistic fuzzy rule-based systems: An overview of interpretability measures. *Information Sciences*, 181(20), 4340–4360.
- Hagras, H. (2018). Toward human-understandable, explainable AI. *Computer*, 51(9), 28–36.
- Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies*, 7(1), 1–13.
- Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.).
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2018). Anchors: High-precision model-agnostic explanations. *AAAI*.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- Smith, J. W., et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proc. Symposium on Computer Applications in Medical Care*, 261–265.
- Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. *IEEE Transactions on Systems, Man, and Cybernetics*, 15(1), 116–132.
- Wang, L.-X., & Mendel, J. M. (1992). Generating fuzzy rules by learning from examples. *IEEE Transactions on Systems, Man, and Cybernetics*, 22(6), 1414–1427.
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.

[Next: Chapter 5 · Architecture →](05-architecture.md)
