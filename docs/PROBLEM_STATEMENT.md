# Glass-Box Insight: Fuzzy Rule Extraction for Explaining Black-Box Neural Network Decisions

## Problem statement

Neural networks often predict well but cannot say *why* they made a decision.
In medicine and finance an unexplained "yes" or "no" is hard to trust, audit
or contest. Fuzzy inference systems have the opposite profile: their IF-THEN
rules use words such as "Glucose is High" that a domain expert can read, but
on their own they struggle to learn complex decision boundaries from data.

This project combines the two. A neural network is trained as the decision
model and then treated as a black box. A fuzzy inference system is trained to
**imitate the network's behaviour** - not the true labels - using a small set
of linguistic rules. The fuzzy system is not meant to be more accurate than
the network; it is meant to be a faithful, readable description of it.

### Formal definition

Given a trained network $f_{NN}: X \to [0,1]$ that outputs P(diabetic), build a
fuzzy system $f_{FIS}: X \to [0,1]$ such that

1. **Fidelity** - $f_{FIS}$ makes the same decision as $f_{NN}$ on unseen inputs
   as often as possible;
2. **Interpretability** - the rule base is small and every linguistic term keeps
   its meaning ("Low" really is low);
3. the **trade-off** between the two is measured over many rule-base sizes.

## Objectives

1. Train and tune a feedforward neural network (backpropagation) on the Pima
   Indians Diabetes dataset as the black-box reference model.
2. Implement a fuzzy rule extraction pipeline: Fuzzy C-Means clustering to
   propose rules, and ANFIS with hybrid learning to tune them to the network.
3. Measure fidelity and interpretability, and plot the trade-off across rule-base
   sizes against a decision-tree surrogate baseline.
4. Explain individual predictions by tracing which rules fired and how much each
   contributed, and check the rules against medical intuition.
5. Assess the value of each design choice with ablation studies (oracle sampling,
   interpretability constraints, imitating the network vs. the true labels).

## Scope

- Dataset: Pima Indians Diabetes (768 patients, 8 numeric features).
- Software only: Python, NumPy, scikit-learn, Matplotlib, Streamlit.
- ANFIS, Fuzzy C-Means, rule pruning and explanations are implemented from
  scratch so that every step can be explained.
- This is a course project: it is not a medical tool and makes no clinical claims.

## Related work

- **Rule extraction from neural networks** has been studied since the 1990s.
  Andrews, Diederich & Tickle (1995) survey the early methods. TREPAN
  (Craven & Shavlik, 1996) introduced the idea of querying the network as an
  oracle to generate extra training data for the surrogate; this project uses
  the same idea for its fuzzy surrogate.
- **ANFIS** (Jang, 1993) represents a Takagi-Sugeno fuzzy system as a five-layer
  network trained by a hybrid of least squares and gradient descent.
- **Fuzzy C-Means** (Bezdek, 1981) is the standard soft-clustering algorithm for
  generating fuzzy rules from data; subtractive clustering (Chiu, 1994) is a
  common alternative.
- **Interpretability of fuzzy systems**: tuning can destroy the meaning of
  linguistic terms; Gacto, Alcalá & Herrera (2011) review measures and
  constraints used to prevent this.
- **Global surrogate models** such as shallow decision trees are a standard XAI
  baseline (Molnar, *Interpretable Machine Learning*).

## Expected outcomes

- A trained black-box network with documented performance.
- A fuzzy rule base that reproduces the network's decisions with high fidelity
  using a handful of short rules.
- A fidelity-vs-interpretability trade-off plot and ablation tables.
- Case studies that explain individual predictions rule by rule.
- An interactive demo for the viva.

## References

1. Jang, J.-S. R. (1993). ANFIS: Adaptive-network-based fuzzy inference system.
   *IEEE Trans. Systems, Man, and Cybernetics*, 23(3), 665-685.
2. Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms*. Plenum.
3. Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications
   to modeling and control. *IEEE Trans. SMC*, 15(1), 116-132.
4. Andrews, R., Diederich, J., & Tickle, A. B. (1995). Survey and critique of techniques
   for extracting rules from trained artificial neural networks. *Knowledge-Based Systems*, 8(6), 373-389.
5. Craven, M., & Shavlik, J. (1996). Extracting tree-structured representations of trained
   networks. *NeurIPS 8*.
6. Chiu, S. L. (1994). Fuzzy model identification based on cluster estimation.
   *Journal of Intelligent & Fuzzy Systems*, 2(3), 267-278.
7. Gacto, M. J., Alcalá, R., & Herrera, F. (2011). Interpretability of linguistic fuzzy
   rule-based systems: An overview of interpretability measures. *Information Sciences*, 181(20), 4340-4360.
8. Smith, J. W., et al. (1988). Using the ADAP learning algorithm to forecast the onset of
   diabetes mellitus. *Proc. Symposium on Computer Applications in Medical Care*, 261-265.
9. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
10. Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). christophm.github.io/interpretable-ml-book
11. Casillas, J., Cordón, O., Herrera, F., & Magdalena, L. (Eds.) (2003). *Interpretability Issues in
    Fuzzy Modeling*. Springer.
12. Hagras, H. (2018). Toward human-understandable, explainable AI. *Computer*, 51(9), 28-36.
