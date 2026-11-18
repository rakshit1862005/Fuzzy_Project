# Explaining Neural Network Decisions in Diabetes Diagnosis Using Fuzzy Logic Rules
### Complete Project Explainer & Web Dashboard User Guide

> **Project Title:** Explaining Neural Network Decisions in Diabetes Diagnosis Using Fuzzy Logic Rules  
> **Course:** Fuzzy Logic and Neural Networks (EEO1003)  
> **Department:** Department of Electrical Engineering, Punjab Engineering College (Deemed to be University), Chandigarh  
> **Mentors:** Dr. Rintu Khanna (Professor) & Dr. Hemant Kumar (Assistant Professor) (Department of Electrical Engineering)  
> **Authors:** Rakshit (23106017), Aarav (23106023), Aryan Garg (23106024), Amish Bansal (23103087)  
> **Academic Year:** 2026--2027  
> **Application Repository:** `d:\Fuzzy_Project\Fuzzy_Project`  
> **Web Dashboard URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Table of Contents
1. [Executive Summary: What is Glass-Box Insight?](#1-executive-summary-what-is-glass-box-insight)
2. [The Core Problem: Why is this Project Needed?](#2-the-core-problem-why-is-this-project-needed)
3. [The Dataset: Clinical Context & Preprocessing Architecture](#3-the-dataset-clinical-context--preprocessing-architecture)
4. [The Core Techniques Used: Step-by-Step Technical Breakdown](#4-the-core-techniques-used-step-by-step-technical-breakdown)
5. [How to Launch and Run the Project](#5-how-to-launch-and-run-the-project)
6. [Complete Web Dashboard Walkthrough (What Every Element Means)](#6-complete-web-dashboard-walkthrough-what-every-element-means)
   - [Topbar & Global Controls](#topbar--global-controls)
   - [View 1: Train (Settings, Stepper, & Real-Time Telemetry)](#view-1-train-settings-stepper--real-time-telemetry)
   - [View 2: Explore (Clinical Sliders, Dual Gauges, & Traceability)](#view-2-explore-clinical-sliders-dual-gauges--traceability)
   - [View 3: Edit Rules (Human-in-the-Loop Counterfactual Simulation)](#view-3-edit-rules-human-in-the-loop-counterfactual-simulation)
   - [View 4: Compare Runs (Pareto Frontier & Historical Benchmarks)](#view-4-compare-runs-pareto-frontier--historical-benchmarks)
7. [Frequently Asked Questions & Professor Viva Defense Guide](#7-frequently-asked-questions--professor-viva-defense-guide)

---

## 1. Executive Summary: What is Glass-Box Insight?

**Glass-Box Insight** is an Explainable Artificial Intelligence (XAI) system designed for clinical decision support. Its core philosophy is summarized by its tagline:
$$\textbf{\textit{``A neural network decides. Fuzzy rules explain why.''}}$$

### The High-Level Concept
- In modern medicine, deep learning models and multi-layer neural networks achieve high diagnostic accuracy, but they operate as **opaque "black boxes"**. They output a single probability number (e.g., *"Patient #152 has an 84% probability of diabetes"*), but they cannot tell the doctor *why*.
- **Glass-Box Insight** solves this problem by training a transparent **Takagi-Sugeno-Kang (TSK) Neuro-Fuzzy Inference System** (ANFIS) to act as a **pedagogical surrogate** that imitates the neural network.
- Instead of uninterpretable weight matrices, our system extracts **4 human-readable IF-THEN rules** with smooth linguistic terms (*Low*, *Medium*, *High*), such as:
  $$\textbf{IF } \text{Glucose is High } \textbf{AND } \text{BMI is High } \textbf{THEN } \text{Risk} = 0.85 \text{ (High Risk)}$$
- The extracted rules achieve **92.2% test fidelity** (and **98.3% confident fidelity**) relative to the neural network, meaning a physician gets the high accuracy of neural computation alongside the complete safety, auditability, and trust of transparent fuzzy logic.

---

## 2. The Core Problem: Why is this Project Needed?

### 2.1 The High Accuracy vs. Dangerous Opacity Dilemma
Artificial Neural Networks (ANNs) excel at finding non-linear patterns in patient records. However, in critical healthcare applications, deploying an opaque model introduces severe risks:
1. **Verification Necessity:** A doctor cannot prescribe aggressive diabetes medication or order expensive biopsies simply because a computer said so without knowing which clinical biomarkers drove the diagnosis.
2. **Confounding Artifacts:** Black boxes can exploit dataset biases, laboratory recording errors, or spurious correlations while maintaining high test accuracy.
3. **Medical Ethics & Doctor-Patient Communication:** Physicians are legally and ethically obligated to explain diagnostic decisions in plain clinical terms to patients.
4. **Regulatory Mandates:**
   - **EU AI Act:** Formally classifies medical AI diagnostic tools as *High-Risk*, mandating human interpretability and traceable decision oversight.
   - **GDPR (Articles 13–22):** Guarantees patients the legal *"Right to Explanation"* for automated decisions.
   - **U.S. FDA SaMD Guidelines:** Mandates full audit trails and decision traceability for Software as a Medical Device.

### 2.2 Why Existing Explainability Tools (LIME, SHAP, Decision Trees) Fall Short
Prior approaches attempted to explain neural networks, but each has major clinical flaws:
- **LIME (Local Interpretable Model-agnostic Explanations):** Perturbs input values randomly and fits local linear models. Because it relies on stochastic sampling, explanations fluctuate wildly across runs for the exact same patient. Furthermore, it only explains a single patient and cannot provide an auditable hospital-wide rule policy.
- **SHAP (Shapley Additive Explanations):** Provides local feature attribution bars, but assumes feature independence (which fails in human physiology where Glucose and Insulin are tightly correlated) and cannot provide global IF-THEN clinical rules.
- **Decision Tree Surrogates (TREPAN):** Uses rigid step-function thresholds (e.g., $\text{Glucose} \le 127.5$). A patient with Glucose = 127.4 mg/dL is classified as completely healthy, while 127.6 mg/dL triggers a diabetic branch—an unrealistic, discontinuous cliff that contradicts continuous human physiology.

### 2.3 The Glass-Box Solution
By bridging **Fuzzy Set Theory** (continuous, graded membership without cliffs) and **Neural Networks** (adaptive gradient learning), Glass-Box Insight provides:
- Smooth, realistic physiological boundaries.
- Global, auditable IF-THEN rule policies.
- Full whiteboard arithmetic traceability (every calculation can be verified by hand in 30 seconds).
- Active detection of out-of-distribution / atypical patients to prevent silent AI hallucinations.

---

## 3. The Dataset: Clinical Context & Preprocessing Architecture

The system is evaluated on the benchmark **Pima Indians Diabetes Database (PIDD)**, curated by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and hosted by the UCI Machine Learning Repository.

### 3.1 Clinical Features and Physiological Axes
The dataset contains records for $N = 768$ female patients of Pima Indian heritage aged 21 and older, categorized across three physiological metabolic axes:

| Physiological Axis | Feature Name | Physical Units | Clinical / Pathological Meaning |
| :--- | :--- | :---: | :--- |
| **Glycemic Control** | `Glucose` | mg/dL | 2-hour plasma glucose concentration in an oral glucose tolerance test (OGTT). Primary diagnostic criterion for diabetes. |
| | `Insulin` | $\mu\text{U/mL}$ | 2-hour post-prandial serum insulin level. Reflects pancreatic beta-cell secretion capacity. |
| **Adiposity & Body Mass** | `BMI` | $\text{kg/m}^2$ | Body Mass Index (weight in kg / $\text{height}^2$ in m). Key indicator of insulin resistance. |
| | `SkinThickness` | mm | Triceps skinfold thickness. Measures subcutaneous adipose tissue deposits. |
| **Demographic & Hereditary** | `Pregnancies` | Count | Number of completed gestational cycles. Gestational diabetes stress marker. |
| | `Age` | Years | Patient age at clinical examination. Beta-cell function declines with age. |
| | `BloodPressure` | mm Hg | Diastolic blood pressure. Indicator of vascular resistance and metabolic syndrome. |
| | `Pedigree` | Unitless | Diabetes Pedigree Function (DPF). Scored genetic predisposition based on family history. |
| **Target Label** | `Outcome` | Binary | $0 =$ Non-diabetic ($N=500, 65.1\%$), $1 =$ Diabetic ($N=268, 34.9\%$). |

### 3.2 The "Zero-as-Missing" Phenomenon & Leakage-Free Imputation
In the raw dataset, measurements that were omitted or unrecorded were encoded as `0`. While `Pregnancies = 0` is biologically valid, values such as `Glucose = 0`, `BloodPressure = 0`, or `BMI = 0` represent **biological impossibilities**:
- `Glucose`: 5 zero values ($0.65\%$)
- `BloodPressure`: 35 zero values ($4.56\%$)
- `BMI`: 11 zero values ($1.43\%$)
- `SkinThickness`: 227 zero values ($29.56\%$)
- `Insulin`: 374 zero values ($48.70\%$)

**Preprocessing Pipeline:**
1. **Stratified Split (60/20/20):** Before any imputation or scaling, the dataset is partitioned using fixed random seed 42 into:
   - **Training Set ($\mathcal{D}_{\text{train}}$):** 460 patients (299 healthy, 161 diabetic)
   - **Validation Set ($\mathcal{D}_{\text{val}}$):** 154 patients (100 healthy, 54 diabetic)
   - **Test Set ($\mathcal{D}_{\text{test}}$):** 154 patients (101 healthy, 53 diabetic)
2. **Leakage-Free Imputation:** Medians are calculated **strictly on positive values from the training partition only**:
   - `Glucose`: 117.0 mg/dL
   - `BloodPressure`: 72.0 mm Hg
   - `SkinThickness`: 29.0 mm
   - `Insulin`: 125.0 $\mu\text{U/mL}$
   - `BMI`: 32.3 $\text{kg/m}^2$
   Validation and test missing values are imputed using these exact training medians, preventing data leakage.
3. **Dual Feature Scaling:**
   - **Standardization (Z-score):** $\tilde{x} = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$ for the MLP neural network to ensure smooth gradient descent.
   - **Min-Max Normalization:** $\tilde{x} = (x - x_{\text{min}}) / (x_{\text{max}} - x_{\text{min}}) \in [0, 1]$ for the ANFIS neuro-fuzzy system to establish bounded, interpretable linguistic universes of discourse.
4. **Synthetic Oracle Sampling ($N_{\text{synth}} = 2{,}000$):** We sample 2,000 synthetic patient vectors uniformly within the clinical feature hypercube and query the trained neural network to label them. This populates sparse boundary regions, giving the fuzzy surrogate rich decision-boundary information.

---

## 4. The Core Techniques Used: Step-by-Step Technical Breakdown

```
Raw Patient Records
       │
       ▼
[Stage 1: Train Neural Network (MLP 8-8-1)] ──► Produces Black-Box Predictions (ROC-AUC: 0.797)
       │
       ▼
[Stage 2: Permutation Feature Importance]  ──► Isolates Top 4 Drivers: Glucose, BMI, Age, Pregnancies
       │
       ▼
[Stage 3: Fuzzy C-Means (FCM) Clustering] ──► Discovers 8 Patient Prototypes (Prevents 3^4=81 explosion)
       │
       ▼
[Stage 4: Shouldered Gaussian Memberships]──► Low, Medium, High terms with flat saturation shoulders
       │
       ▼
[Stage 5: 5-Layer TSK ANFIS Architecture]  ──► Fuzzification → T-Norm → Normalize → Consequent → Sum
       │
       ▼
[Stage 6: Hybrid Learning Optimization]   ──► Ridge Least Squares (Consequents) + Adam (Premises)
       │
       ▼
[Stage 7: Semantic Integrity Constraints] ──► Enforces Low < Med < High & Drift bounds (Anti-Collapse)
       │
       ▼
[Stage 8: Greedy Backward Rule Pruning]   ──► Prunes conditions & rules (8 rules → 4 compact rules)
       │
       ▼
[Stage 9: Clinical Deployment & Telemetry]──► 92.2% Test Fidelity, Sub-5ms Live Interactive Web Dashboard
```

### 4.1 Stage 1: The Black-Box Neural Network (StreamingMLP)
- **Architecture:** Feedforward Multi-Layer Perceptron (8 input features $\to$ 8 hidden neurons with ReLU activation $\to$ 1 output neuron with Sigmoid activation).
- **Complexity:** Only 81 trainable parameters ($(8 \times 8 + 8) + (8 \times 1 + 1) = 81$), avoiding severe overfitting on small clinical cohorts.
- **Optimization:** Adam optimizer ($\beta_1=0.9, \beta_2=0.999$, learning rate $\eta=0.01$), regularized binary cross-entropy loss with L2 penalty ($\alpha=1.0$).
- **Performance:** 70.1% test accuracy, 0.797 ROC-AUC, 5-fold CV score of $0.826 \pm 0.017$.

### 4.2 Stage 2: Permutation Feature Importance (PFI)
If we fed all 8 features into a fuzzy system with 3 linguistic terms each, grid partitioning would generate $3^8 = \mathbf{6{,}561 \text{ rules}}$—an impossible cognitive burden.
To reduce dimensionality principledly, we measure how much the neural network's cross-entropy loss degrades when each feature column is randomly shuffled:
1. **Glucose:** 43.8% importance share (Loss increase +0.231) — **Selected (Rank 1)**
2. **BMI:** 20.8% importance share (Loss increase +0.110) — **Selected (Rank 2)**
3. **Age:** 9.7% importance share (Loss increase +0.051) — **Selected (Rank 3)**
4. **Pregnancies:** 8.6% importance share (Loss increase +0.045) — **Selected (Rank 4)**
5. *Non-selected features:* BloodPressure (5.9%), Insulin (4.8%), SkinThickness (3.4%), Pedigree (3.0%).
Selecting the top 4 salient features captures **82.9% of the network's predictive mass**, while shrinking the input dimensionality from 8 down to 4.

### 4.3 Stage 3: Data-Driven Fuzzy C-Means (FCM) Clustering
Even with 4 features, full grid partitioning produces $3^4 = \mathbf{81 \text{ rules}}$.
To solve this, we run **Fuzzy C-Means clustering** (Bezdek, 1981) with fuzziness exponent $m=2.0$ on the 4-dimensional normalized patient data. FCM discovers $C=8$ multi-dimensional cluster centroids (patient prototypes) that reflect actual clinical patterns, directly initializing **8 candidate rules**.

### 4.4 Stage 4: Shouldered Gaussian Membership Functions
Standard Gaussian curves decay to zero at extremes. For a patient with extreme glucose ($\ge 200$ mg/dL), a standard Gaussian would output near-zero membership, leaving the patient with no firing rules!
We implement **monotonic flat boundary shoulders**:
- **`Low`:** Gaussian curve for $x \ge c_{\text{Low}}$, but remains **flat at 1.0** for all $x \le c_{\text{Low}}$.
- **`Medium`:** Standard bell-shaped Gaussian centered at $c_{\text{Med}}$.
- **`High`:** Gaussian curve for $x \le c_{\text{High}}$, but remains **flat at 1.0** for all $x \ge c_{\text{High}}$.
This guarantees that severely ill patients always receive full membership in *High*, completely eliminating unhandled edge cases.

### 4.5 Stage 5: Five-Layer Takagi-Sugeno-Kang (TSK) ANFIS Architecture
The extracted system is executed through 5 transparent feedforward layers:
- **Layer 1 (Fuzzification):** Computes continuous membership grades $\mu_{jt}(x_j) \in [0, 1]$ for each feature in *Low*, *Medium*, or *High*.
- **Layer 2 (Rule Firing via Product T-Norm):** Multiplies memberships across the active conditions of rule $r$:
  $$w_r = \prod_{j \in \text{Cond}(r)} \mu_{j, A_{rj}}(x_j)$$
  We use the differentiable Product operator ($\times$) rather than the Min ($\min$) operator, ensuring smooth gradient propagation.
- **Layer 3 (Normalization):** Calculates the relative firing weight of each rule:
  $$\bar{w}_r = \frac{w_r}{\sum_{k=1}^R w_k}, \quad \text{such that } \sum_{r=1}^R \bar{w}_r = 1.0$$
- **Layer 4 (Rule Consequents):** In zero-order TSK, each rule outputs a fixed risk scalar $p_r \in [0, 1]$.
- **Layer 5 (Defuzzified Output):** Calculates the final patient diabetes risk score as a weighted sum:
  $$y = \sum_{r=1}^R \bar{w}_r \cdot p_r$$

### 4.6 Stage 6: Hybrid Learning Optimization
ANFIS updates its parameters through a fast, stable two-step hybrid algorithm:
1. **Forward Pass (Least Squares for Consequents):** With Gaussian membership curves held fixed, the rule output risk weights $\mathbf{p} = [p_1, \dots, p_R]^T$ are solved in **a single mathematical step** using Ridge Regularized Least Squares (RLS):
   $$\mathbf{p}^* = (\bar{\mathbf{W}}^T \bar{\mathbf{W}} + \lambda \mathbf{I})^{-1} \bar{\mathbf{W}}^T \mathbf{t}$$
   where $\mathbf{t}$ contains the continuous probabilities output by the neural network.
2. **Backward Pass (Gradient Descent for Premises):** With consequents $\mathbf{p}$ fixed, prediction errors $(y_i - t_i)$ are backpropagated to update the Gaussian centers ($c$) and widths ($\sigma$) using the Adam optimizer ($\eta=0.01$).

### 4.7 Stage 7: Semantic Integrity Preservation Constraints
Unconstrained gradient descent can cause **semantic collapse**—where the *Low* center drifts higher than the *High* center, or curves shrink into needle-thin spikes.
After every training iteration, three geometric constraints are enforced:
1. **Strict Term Ordering:** $c_{\text{Low}} < c_{\text{Med}} < c_{\text{High}}$ (enforced with a minimum gap of $0.05$).
2. **Bounded Anchor Drift:** Centers cannot drift more than $\pm 0.10$ from their original data-derived cluster centers ($|c - c^{(0)}| \le 0.10$).
3. **Minimum Width Floor:** Gaussian widths are bounded below by $\sigma \ge 0.05$ to prevent delta-function spikes.

### 4.8 Stage 8: Greedy Backward Rule Pruning
The initialized 8-rule system contains redundant conditions (e.g., if a rule fires whenever Glucose is High regardless of Age, Age is redundant).
We run an automated **two-check greedy backward elimination** algorithm:
1. Iteratively test deleting each condition one by one.
2. An edit is accepted if and only if:
   - Holdout validation fidelity drops by less than the user tolerance ($\Delta \mathcal{F}_{\text{val}} \le 0.01$).
   - Absolute patient probability error remains bounded ($\text{MAE} \le 0.05$).
3. If all conditions in a rule become empty, the entire rule is pruned.
**Result:** The rule base drops from **8 rules and 32 conditions** down to **4 compact rules and 12 conditions**—a **96% complexity reduction** from the theoretical grid while retaining **92.2% test fidelity**.

### 4.9 Stage 9: Safety Diagnostics & Traceability
- **Hand Traceability:** The exact calculation for any patient can be computed on a whiteboard in 30 seconds using standard arithmetic.
- **Weak-Match / Out-of-Distribution Diagnostic Alert:** The maximum raw firing strength is tracked:
  $$\mathcal{S}_{\text{match}} = \max_{r=1 \dots R} w_r$$
  If an atypical patient arrives whose clinical profile does not match any known prototype (e.g., an elderly, severely malnourished patient with extreme glucose), $\mathcal{S}_{\text{match}}$ drops below $0.20$. The system immediately triggers an amber/red **"Weak Match: Out of Distribution"** alert, warning doctors that the automated system cannot reliably evaluate this patient—preventing silent AI errors.

---

## 5. How to Launch and Run the Project

### 5.1 System Requirements & Verification
The project runs on standard Python 3.10+ (tested on Python 3.13) with standard scientific computing packages:
- `fastapi`, `uvicorn`, `starlette` (Web server & REST API)
- `numpy`, `pandas`, `scipy` (Data handling & linear algebra)
- `pytest` (Automated testing suite)

### 5.2 Starting the Web Server
Open PowerShell or your command terminal and navigate to the project root:
```powershell
cd d:\Fuzzy_Project\Fuzzy_Project
uvicorn webapp.server:app --host 127.0.0.1 --port 8000
```
When successful, the terminal displays:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 5.3 Opening the Dashboard
Open your browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 5.4 Running the Automated Test Suite
To verify the integrity of all algorithmic modules, execute pytest:
```powershell
cd d:\Fuzzy_Project\Fuzzy_Project
pytest tests/
```
All **26 tests** will pass in ~6 seconds:
- `test_data.py`: Verifies stratified splits, median imputation, and normalization.
- `test_blackbox.py`: Verifies MLP forward pass and permutation feature importance.
- `test_fcm.py`: Verifies Fuzzy C-Means convergence and membership constraints.
- `test_anfis.py`: Verifies 5-layer forward propagation, Ridge RLS, and hybrid learning.
- `test_metrics_and_rules.py`: Verifies greedy pruning and fidelity metrics.
- `test_webapp.py`: Verifies FastAPI REST endpoints and SSE telemetry streams.

---

## 6. Complete Web Dashboard Walkthrough (What Every Element Means)

The web dashboard is divided into four main functional views accessible via the top navigation bar.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  GLASS-BOX INSIGHT     [1 · Train] [2 · Explore] [3 · Edit rules] [4 · Runs] │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### Topbar & Global Controls

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ [Logo] Glass-Box Insight                         [Run #1 · 4 rules]  [ ◐ ]   │
│        A neural network decides. Fuzzy rules explain why.                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

1. **Brand & Tagline:** Identifies the system and its pedagogical neuro-fuzzy surrogate role.
2. **Navigation Tabs:**
   - `1 · Train`: Configure models and watch the complete training pipeline run live.
   - `2 · Explore`: Interactive clinical sliders and real-time dual-model patient diagnosis.
   - `3 · Edit rules`: Live interactive rule editor for human-in-the-loop counterfactual simulation.
   - `4 · Compare runs`: Pareto frontier scatter chart and historical run logs.
3. **Run Pill (`#run-pill`):**
   - Shows the status of the currently loaded model (e.g., `No run yet` or `Run #1 · 4 rules · 92.2% fidelity`).
   - Turns green on successful training completion or yellow while training is actively streaming.
4. **Theme Toggle (`◐`):** Instantly toggles between high-contrast Dark Mode and clean Clinical Light Mode.

---

### View 1: Train (Settings, Stepper, & Real-Time Telemetry)

This view allows you to configure hyper-parameters, initiate training, and watch the system learn in real time via Server-Sent Events (SSE).

```
┌───────────────────────────┬──────────────────────────────────────────────────┐
│  SETTINGS                 │  [Data]─►[Network]─►[Importance]─►[ANFIS]─►[Prune]│
│                           ├─────────────────────────┬────────────────────────┤
│  Black Box Neural Network │ ① Neural Network        │ ② Feature Importance   │
│  - Hidden layers: 8       │   [Loss Decay Chart]    │   [PFI Bar Chart]      │
│  - L2 penalty α: 1        ├─────────────────────────┴────────────────────────┤
│                           │ ③ ANFIS Learns to Imitate Network                │
│  Fuzzy Explainer (ANFIS)  │   [Hybrid Loss Chart] | [Membership Curves Low/H]│
│  - Inputs: 4              ├──────────────────────────────────────────────────┤
│  - Terms: 3 (Low/Med/High)│ ④ Rule Base: Proposed by FCM, Then Pruned        │
│  - Clusters: 8            │   [Active Rules List] | [Pruning Event Log]      │
│  - Target: The network    ├──────────────────────────────────────────────────┤
│                           │ ⑤ Evaluation Results on 154 Test Patients        │
│  [ ▶ Train ]  [ Reset ]   │   [ 92.2% Fidelity ] [ 70.1% Acc ] [ 0.797 AUC ] │
└───────────────────────────┴──────────────────────────────────────────────────┘
```

#### The Settings Form (Left Sidebar)
- **Black box · neural network:**
  - `Hidden layers`: Selects the network depth (`8 neurons` (default), `16 neurons`, `16 → 8 neurons`, or `32 → 16 neurons`).
  - `L2 penalty α`: Weight decay regularization ($0.001, 0.01, 0.1, 1.0$). Higher $\alpha$ prevents overfitting.
  - `Choose size and α by 5-fold cross-validation`: Checkbox that runs an automated grid search over architectures before training.
- **Fuzzy explainer · ANFIS:**
  - `Inputs used by the rules` (Slider 2 to 6): Controls how many salient features from PFI are passed to the fuzzy explainer (Default: 4).
  - `Terms per input`: Choose between 2 linguistic terms (*Low*, *High*) or 3 terms (*Low*, *Medium*, *High*).
  - `FCM clusters` (Slider 2 to 16): Upper bound on initial candidate rules (Default: 8).
  - `THEN-part`: Choose between **Zero-order TSK** (Constant scalar, e.g., $\text{Risk} = 0.85$) or **First-order TSK** (Linear polynomial formula).
  - `Training epochs` (Slider 20 to 600): Number of hybrid learning cycles (Default: 100).
  - `Imitate`:
    - `The network (explainer)`: True surrogate mode. ANFIS imitates the neural network's continuous probabilities.
    - `The true diagnosis (control)`: Trains the fuzzy system directly on ground-truth $\{0, 1\}$ clinical labels (serves as an experimental control baseline).
  - `Keep terms meaningful (constraints)`: Enforces ordering ($c_{\text{Low}} < c_{\text{Med}} < c_{\text{High}}$) and prevents semantic collapse.
- **Data & simplification:**
  - `Synthetic patients asked to the network` (0 to 4,000): Number of synthetic oracle neighborhood queries used to populate boundary regions (Default: 2,000).
  - `Prune conditions and rules`: Enables the automated greedy backward pruning engine.
  - `Allowed fidelity loss while pruning` ($0\%$ to $5\%$): Maximum permitted validation fidelity drop per pruning step (Default: $1\%$).
  - `Random seed`: Controls synthetic sampling and cluster initialization (Default: 42).
  - `Slow motion`: Adds an artificial 25ms delay per epoch so users can visually watch the curves move live on projector screens.
- **Action Buttons:**
  - `▶ Train`: Starts the asynchronous training pipeline.
  - `Report settings`: Restores all parameters to the exact benchmark configuration reported in the academic paper.

#### The 7-Stage Pipeline Stepper
As training executes, the horizontal stepper highlights the active algorithmic stage:
1. `Data`: Ingestion, stratified splitting, and median imputation.
2. `Neural network`: Backpropagation training of the MLP.
3. `Importance`: Permutation feature importance shuffling.
4. `FCM rules`: Unsupervised prototype clustering.
5. `ANFIS`: Hybrid optimization of premise curves and consequent weights.
6. `Pruning`: Backward greedy elimination of redundant conditions.
7. `Evaluation`: Final benchmarking across the 154 holdout test patients.

#### Live Telemetry Dashboard Cards
- **Card ① Neural Network Learns:**
  - Displays the live training loss curve (Binary Cross-Entropy) decreasing across epochs.
  - KPI badge shows final training loss and convergence epoch.
- **Card ② What the Network Relies On:**
  - Horizontal bar chart showing the permutation importance share of each clinical feature.
  - The top features selected for the fuzzy explainer are highlighted in bright blue.
- **Card ③ ANFIS Learns to Imitate the Network:**
  - **Left:** Mean Squared Error (MSE) loss decay curve as ANFIS imitates the neural network's output probabilities.
  - **Right (MF Grid):** Real-time animated Gaussian membership curves for Glucose, BMI, Age, and Pregnancies. Dashed curves represent the initial cluster positions; solid lines show the curves being fine-tuned live by gradient descent.
- **Card ④ Rule Base: Proposed by FCM, Then Pruned:**
  - **Left:** The live rule list. Rules that get completely pruned disappear from this list.
  - **Right (Pruning Event Log):** Real-time text log of every accepted pruning decision (e.g., `Pruned Age is High from Rule 3 (Fidelity: 94.8% -> 94.6%)`).
- **Card ⑤ Evaluation Results:**
  - Unrolls once training finishes, presenting holdout test metrics:
    - **Fidelity:** Percentage of test patients where ANFIS and the Neural Network assign the identical diagnostic class ($92.2\%$).
    - **NN Test Accuracy:** Diagnostic accuracy of the black box ($70.1\%$).
    - **Surrogate Accuracy:** Diagnostic accuracy of the fuzzy explainer ($72.1\%$).
    - **ROC-AUC:** Area under the ROC curve ($0.797$).
  - Action buttons allow instant transition to **"Explore patients →"** or **"Edit the rules →"**.

---

### View 2: Explore (Clinical Sliders, Dual Gauges, & Traceability)

The Explore view is the primary clinical auditing workstation where a physician can evaluate real patients or test hypothetical "what-if" scenarios.

```
┌───────────────────────┬──────────────────────────────────────────────────────┐
│  PATIENT SELECTION    │  CONCURRENT DUAL VERDICT                             │
│  [ Patient #152    ▼] │   [ Black Box MLP ]         [ Glass-Box ANFIS ]      │
│                       │      83.7% Risk                 84.3% Risk           │
│  CLINICAL SLIDERS     │     (Diabetic)                 (Diabetic)            │
│  Glucose:    197 mg/dL│   [Badge: Perfect Agreement]  [Badge: Strong Match]  │
│  BMI:        34.7     ├──────────────────────────────────────────────────────┤
│  Age:        31 yrs   │  WHY: THE RULES THAT FIRED                           │
│  Pregnancies: 2       │  Stacked Bar: [██████████████████████░░░░] 84.3% Risk│
│                       │  Rule 4 (Diabetic):  Weight 98.4% | Contribution 0.84│
│  [True Label: 1]      │  Rule 1 (Healthy):   Weight  1.6% | Contribution 0.00│
│                       ├──────────────────────────────────────────────────────┤
│                       │  FUZZIFICATION BREAKDOWN                             │
│                       │  Glucose (197): Low 0.00 | Med 0.00 | High 1.00      │
│                       │  BMI (34.7):    Low 0.00 | Med 0.65 | High 0.35      │
└───────────────────────┴──────────────────────────────────────────────────────┘
```

#### 1. Patient Selector & Input Sliders (Left Sidebar)
- **Start from Dropdown:** Allows selecting any of the 154 holdout test patients or predefined clinical prototypes:
  - `Diabetic Prototype`: Severe hyperglycemia and high BMI.
  - `Healthy Prototype`: Low glucose, lean BMI, young age.
  - `Boundary Ambiguity Case`: Borderline glucose (125 mg/dL) near clinical diagnostic thresholds.
  - `Atypical Outlier Case`: Severe hyperglycemia in an elderly patient with lean BMI (triggers weak match).
- **Interactive Sliders:**
  - `Glucose` ($0$ to $200$ mg/dL, step $1$)
  - `BMI` ($15.0$ to $60.0 \text{ kg/m}^2$, step $0.1$)
  - `Age` ($21$ to $85$ years, step $1$)
  - `Pregnancies` ($0$ to $17$, step $1$)
- Whenever a slider is dragged, an asynchronous call is dispatched to the Python backend; predictions, gauges, and rule activations update in **less than 5 milliseconds**.

#### 2. Concurrent Dual Verdict Gauges
- **Gauge 1 (Neural Network):** Displays the continuous probability $P_{\text{NN}} \in [0\%, 100\%]$ produced by the multi-layer perceptron.
- **Gauge 2 (Glass-Box Fuzzy):** Displays the exact defuzzified probability $P_{\text{fuzzy}} \in [0\%, 100\%]$ produced by the ANFIS surrogate.
- **Diagnostic Status Badges:**
  - `Agreement Badge`: Shows `Agreement` (both predict healthy or both predict diabetic) or `Disagreement` (if the models cross the 0.50 boundary).
  - `Match-Strength Badge`: Shows `Strong Match` ($\mathcal{S}_{\text{match}} \ge 0.50$), `Moderate Match`, or `Weak Match: Out of Distribution` ($\mathcal{S}_{\text{match}} < 0.20$, highlighted in amber/red).

#### 3. Card "Why: The Rules That Fired"
This panel provides complete, transparent decision decomposition:
- **Stacked Probability Bar:** Visualizes total risk. Green segments represent contributions from non-diabetic rules; red segments represent contributions from diabetic rules.
- **Fired Rules Breakdown Table:**
  - **Rule Sentence:** Full linguistic text (e.g., `IF Glucose is High AND BMI is Medium THEN Risk = 0.85`).
  - **Raw Firing Strength ($w_r$):** The product of membership grades across active conditions.
  - **Normalized Weight ($\bar{w}_r$):** The rule's percentage share of the overall clinical decision ($\sum \bar{w}_r = 100\%$).
  - **Consequent Risk ($p_r$):** The base risk score associated with this rule prototype.
  - **Risk Contribution:** Exactly equals $\bar{w}_r \times p_r$. The sum of all contributions **adds up exactly to the final fuzzy risk percentage**.

#### 4. Card "Fuzzification Breakdown"
Visualizes exactly how the patient's raw clinical measurements map into linguistic terms:
- Displays 4 plots (one for each feature) with *Low*, *Medium*, and *High* curves.
- A vertical indicator line marks the patient's current slider value.
- Clear numeric tags show the exact membership grade (e.g., $\mu_{\text{High}}(\text{Glucose}) = 0.98$).

---

### View 3: Edit Rules (Human-in-the-Loop Counterfactual Simulation)

The Edit Rules view allows clinical domain experts to inspect, modify, add, or delete rules, and observe the immediate mathematical impact on system fidelity.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EDIT THE RULE BASE                 [ + Add Rule ]  [ Reset to Trained ]     │
│  Holdout Test Fidelity: 92.2% | Conditions: 12 | Rules: 4                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  RULE 1 (Healthy Baseline)                                                   │
│  IF [Glucose is Low ▾] [×] AND [BMI is Low ▾] [×]     THEN Risk: 0.08 [Save] │
│  [ + Add condition ]                                            [Delete Rule]│
├──────────────────────────────────────────────────────────────────────────────┤
│  RULE 4 (Overt Diabetes)                                                     │
│  IF [Glucose is High ▾] [×] AND [BMI is High ▾] [×]    THEN Risk: 0.85 [Save] │
│  [ + Add condition ]                                            [Delete Rule]│
└──────────────────────────────────────────────────────────────────────────────┘
```

#### How to Interact with the Rule Editor
1. **Change a Condition:** Click the dropdown on any condition pill (e.g., change `Glucose is High` to `Glucose is Medium`).
2. **Delete a Condition:** Click the small `×` icon next to a condition to remove it from the rule.
3. **Add a Condition:** Click `+ Add condition` to introduce an additional biomarker constraint to an existing rule.
4. **Delete a Rule:** Click `Delete Rule` to remove an entire rule from the knowledge base.
5. **Add a Rule:** Click `+ Add rule` at the top right to define an entirely new clinical scenario.
6. **Instantaneous Re-Fitting (Ridge RLS):**
   - The moment any rule edit is made, the Python backend executes Ridge Regularized Least Squares across the training cohort in **under 3 milliseconds**.
   - Consequent risk values ($p_r$) re-optimize instantly.
   - The top banner immediately updates the **Test-Set Fidelity** score (e.g., showing if deleting a condition caused fidelity to drop from $92.2\%$ to $89.5\%$).
7. **Reset to Trained Rules:** Click `Reset to trained rules` at any time to undo all manual edits and restore the algorithmically optimized model.

---

### View 4: Compare Runs (Pareto Frontier & Historical Benchmarks)

This view provides comprehensive experimental auditing, allowing researchers and professors to compare multiple runs across different hyper-parameter settings.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PARETO FRONTIER: FIDELITY VS. COMPLEXITY                                    │
│                                                                              │
│   Fidelity (%)                                                               │
│   100% ┼                                                                     │
│        │               ● (Run #1: ANFIS 4 rules, 12 cond, 92.2% fidelity)    │
│    90% ┼         ●     ▲ (TREPAN Decision Tree: 12 cond, 84.1% fidelity)     │
│        │                                                                     │
│    80% ┼   ■ (Direct Fuzzy: 12 cond, 78.4% fidelity)                         │
│        └───────┬──────────────┬──────────────┬──────────────┬────────────►   │
│                5             10             15             20  Conditions    │
├──────────────────────────────────────────────────────────────────────────────┤
│  RUNS AUDIT TABLE                                                            │
│  Run ID   | Model Type | Rules | Conditions | Test Fidelity | ROC-AUC | Date │
│  Run #1   | ANFIS (NN) |   4   |     12     |     92.2%     |  0.797  | ...  │
│  Run #2   | Tree-Surr  |   6   |     18     |     85.7%     |  0.781  | ...  │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Elements of the Compare View
1. **The Pareto Frontier Scatter Chart:**
   - **Y-Axis:** Holdout Test-Set Fidelity ($\%$ agreement with the Neural Network).
   - **X-Axis:** Model Complexity (Total number of active conditions across all rules).
   - **Optimal Region (Top-Left):** Represents the ideal clinical trade-off—maximum fidelity with minimum cognitive complexity.
   - **Legend Markers:**
     - **Blue Solid Circle (`●`):** Glass-Box ANFIS surrogate imitating the neural network.
     - **Red Solid Circle (`●`):** Control fuzzy model trained directly on true labels $\{0, 1\}$.
     - **Hollow Circle (`○`):** Model state after manual clinician edits.
     - **Green Triangle (`▲`):** Decision Tree (TREPAN) baseline of equivalent size.
2. **Historical Runs Audit Table:**
   - Displays every run executed during the session with its exact hyper-parameter configuration.
   - Clicking on any historical row immediately reloads that model into memory, allowing you to re-explore its patients in View 2 or inspect its rules in View 3.

---

## 7. Frequently Asked Questions & Professor Viva Defense Guide

### Q1: Is this project strictly within the syllabus of our course (EEO1003)?
**Yes, 100%.** Every algorithmic component directly implements foundational topics from the course domain:
- **Neural Networks:** Multi-Layer Perceptrons, backpropagation learning, binary cross-entropy, and gradient optimization (grounded in Simon Haykin and Laurene Fausett).
- **Fuzzy Logic:** Linguistic variables, continuous membership functions, T-norms, zero-order Takagi-Sugeno-Kang fuzzy inference, and defuzzification (grounded in Timothy J. Ross and S.N. Sivanandam).
- **Neuro-Fuzzy Synthesis:** ANFIS architecture (Jang, 1993) and Fuzzy C-Means clustering (Bezdek, 1981).
- The *only* modern addition is the overarching problem formulation: using the neuro-fuzzy system as a pedagogical surrogate for Explainable AI (XAI) rather than a simple standalone classifier.

### Q2: Why train a Neural Network first if we end up using Fuzzy Rules?
Because neural networks possess superior non-linear gradient search capabilities that find complex decision boundaries in high-dimensional continuous data. A fuzzy system initialized from scratch often gets trapped in local optima or suffers from the curse of dimensionality ($3^8 = 6{,}561$ rules).
By using the **Pedagogical Surrogate Philosophy** (Andrews et al., 1995), the neural network acts as an "oracle teacher", smoothing out noise in the raw dataset. ANFIS then distills that teacher's knowledge into 4 clean, human-auditable rules.

### Q3: Why use Zero-Order TSK instead of Mamdani or First-Order TSK?
- **Against Mamdani:** Mamdani uses fuzzy output sets requiring numerical integration (centroid defuzzification), which is computationally expensive and difficult to optimize with gradient descent.
- **Against First-Order TSK:** First-order TSK uses linear hyperplanes in the THEN-part ($\text{Risk} = a_0 + a_1 x_1 + \dots$). If the THEN-part is an equation, a doctor still cannot interpret the rule at a glance.
- **Zero-Order TSK Advantage:** The THEN-part is a single, intuitive scalar percentage ($\text{Risk} = 0.85$), providing immediate cognitive clarity while remaining 100% differentiable for hybrid learning.

### Q4: Why use the Product T-Norm instead of the Min operator?
The standard Zadeh Min operator ($\min(\mu_1, \mu_2)$) has discontinuous, non-differentiable switching points: $\frac{\partial}{\partial \mu_1} \min(\mu_1, \mu_2) = 0$ whenever $\mu_1 > \mu_2$. This blocks gradient backpropagation to the non-minimum feature.
The **Product T-Norm** ($w_r = \mu_1 \times \mu_2 \times \dots$) is smooth, infinitely differentiable everywhere, and ensures that every condition contributes proportional gradient information during backward premise tuning.

### Q5: How does the system prevent semantic collapse?
In unconstrained ANFIS, gradient descent often causes *Low* and *High* membership curves to invert or collapse into sharp delta spikes.
Our engine enforces **three strict geometric projection steps** after every Adam training step:
1. $c_{\text{Low}} \le c_{\text{Med}} - 0.05$ and $c_{\text{Med}} \le c_{\text{High}} - 0.05$ (ordering constraint).
2. $|c - c^{(0)}| \le 0.10$ (drift bounds from data cluster centers).
3. $\sigma \ge 0.05$ (width floor preventing delta spikes).

### Q6: What happens if an atypical or anomalous patient is tested?
In black-box models, out-of-distribution patients cause **silent catastrophic failures** (the network confidently emits an incorrect risk score).
Glass-Box Insight actively computes the match-strength metric:
$$\mathcal{S}_{\text{match}} = \max_{r=1 \dots R} w_r$$
If $\mathcal{S}_{\text{match}} < 0.20$, the patient falls outside the support of all extracted clinical prototypes. The dashboard immediately triggers an amber/red **"Weak Match: Out of Distribution"** diagnostic flag, warning clinicians to disregard automated scoring and conduct direct manual pathology tests.

---

*Document compiled and verified for the Department of Electrical Engineering, Punjab Engineering College (Deemed to be University), Chandigarh.*

