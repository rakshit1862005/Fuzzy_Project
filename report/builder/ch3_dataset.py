# Chapter 3: Dataset Ingestion, Clinical Attributes and Preprocessing

CHAPTER_3 = r"""
\chapter{Dataset Ingestion, Clinical Attributes and Preprocessing}

\section{The Pima Indians Diabetes Database: Context and Demographics}

The empirical evaluation of the Glass-Box Insight framework is conducted on the benchmark \textbf{Pima Indians Diabetes Database} (PIDD), curated by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and preserved by the UCI Machine Learning Repository (Smith et al., 1988).

The Pima population of the Gila River Indian Community historically exhibited the highest global prevalence of Type 2 diabetes mellitus, exceeding 50\% in adults over 35 years. Epidemiological literature attributes this predisposition to the ``thrifty gene hypothesis''—evolutionary adaptations enabling efficient caloric conservation during famine cycles, which cause rapid metabolic breakdown, insulin resistance, and visceral adiposity in the presence of modern sedentary habits and high-glycemic diets.

The database comprises records from $N = 768$ female patients aged $\ge 21$ years, clinically evaluated per World Health Organization (WHO) protocols. The binary target \texttt{Outcome} records whether diabetes was diagnosed within five years of examination:
\begin{itemize}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Class 0 (Non-diabetic):} 500 patients (65.10\%)
    \item \textbf{Class 1 (Diabetic):} 268 patients (34.90\%)
\end{itemize}
The class imbalance (1.87:1) necessitates stratified sampling and balanced evaluation metrics (such as ROC-AUC and F1-score) to prevent majority-class bias.

\section{Clinical Attributes and Physiological Correlations}

Each patient record contains $d = 8$ physiological and demographic attributes, summarized in Table~\ref{tab:clinical_features}.

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.80}
\begin{tabularx}{\textwidth}{l l X r r r}
\toprule
\textbf{Feature Name} & \textbf{Units} & \textbf{Diagnostic / Pathological Meaning} & \textbf{Min} & \textbf{Median} & \textbf{Max} \\
\midrule
\texttt{Pregnancies} & Count & Number of completed gestational cycles. & 0 & 3.0 & 17 \\
\texttt{Glucose} & mg/dL & 2-hr plasma glucose in an oral glucose tolerance test. & 0 & 117.0 & 199 \\
\texttt{BloodPressure} & mm Hg & Diastolic blood pressure; vascular resistance marker. & 0 & 72.0 & 122 \\
\texttt{SkinThickness} & mm & Triceps skinfold thickness; subcutaneous fat marker. & 0 & 23.0 & 99 \\
\texttt{Insulin} & $\mu\text{U/mL}$ & 2-hr serum insulin; pancreatic beta-cell response. & 0 & 30.5 & 846 \\
\texttt{BMI} & $\text{kg/m}^2$ & Body Mass Index; weight in kg divided by height in $\text{m}^2$. & 0.0 & 32.0 & 67.1 \\
\texttt{Pedigree} & Unitless & Diabetes Pedigree Function; hereditary genetic score. & 0.078 & 0.372 & 2.42 \\
\texttt{Age} & Years & Patient age at clinical examination. & 21 & 29.0 & 81 \\
\bottomrule
\end{tabularx}
\caption[Clinical Attributes and Baseline Statistics in PIDD]{Clinical attributes in the Pima Indians Diabetes Database and baseline statistics before data imputation.}
\label{tab:clinical_features}
\end{table}

These attributes span three principal metabolic axes:
\begin{itemize}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Glycemic Regulation (\texttt{Glucose}, \texttt{Insulin}):} Elevated plasma glucose is the diagnostic hallmark of diabetes. Serum insulin reflects beta-cell compensatory hypersecretion or subsequent secretory failure.
    \item \textbf{Adiposity (\texttt{BMI}, \texttt{SkinThickness}):} Visceral adipose tissue releases pro-inflammatory cytokines that disrupt insulin receptor phosphorylation in peripheral tissues.
    \item \textbf{Genetic and Demographic Risk (\texttt{Age}, \texttt{Pedigree}, \texttt{Pregnancies}):} Age correlates with tissue resistance; pedigree measures familial risk; multiparity is linked to gestational diabetes stress.
\end{itemize}

\section{The Zero-as-Missing Phenomenon: Biological Impossibilities}

A critical vulnerability in raw PIDD data is numerical zeros denoting missing entries. A living patient cannot exhibit zero glucose (fatal hypoglycemia), zero blood pressure (cardiac arrest), zero skinfold thickness, zero serum insulin, or zero BMI. Table~\ref{tab:missing_zeros} quantifies these unrecorded observations.

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.80}
\begin{tabularx}{\textwidth}{l r r X}
\toprule
\textbf{Clinical Feature} & \textbf{Zero Count} & \textbf{Missing \%} & \textbf{Physiological Reality} \\
\midrule
\texttt{Glucose} & 5 & 0.65\% & Laboratory recording omission; essential diagnostic metric. \\
\texttt{BloodPressure} & 35 & 4.56\% & Unmeasured sphygmomanometer reading. \\
\texttt{BMI} & 11 & 1.43\% & Unrecorded patient weight or height. \\
\texttt{SkinThickness} & 227 & 29.56\% & Omitted caliper skinfold measurement. \\
\texttt{Insulin} & 374 & 48.70\% & Expensive 2-hr post-prandial serum assay not administered. \\
\bottomrule
\end{tabularx}
\caption[Quantification of Biologically Impossible Zero Values]{Quantification of biologically impossible zero values representing missing clinical entries.}
\label{tab:missing_zeros}
\end{table}

Nearly 49\% of insulin and 30\% of skinfold values are unrecorded. Leaving these as zeros causes neural networks to learn spurious correlations (e.g., zero insulin falsely predicting low risk).

\section{Leakage-Free Preprocessing: Stratified Split and Median Imputation}

To prevent \textbf{data leakage}, all preprocessing statistics are strictly isolated within the training partition.

\subsection{Stratified Train-Validation-Test Splitting}
The dataset ($N = 768$) is partitioned via stratified random sampling (fixed seed 42) into three disjoint subsets:
\begin{enumerate}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Training Partition ($\mathcal{D}_{\text{train}}$):} 60\% ($N_{\text{train}} = 460$; 299 non-diabetic, 161 diabetic). Used to fit model weights and compute imputation statistics.
    \item \textbf{Validation Partition ($\mathcal{D}_{\text{val}}$):} 20\% ($N_{\text{val}} = 154$; 100 non-diabetic, 54 diabetic). Used to tune hyperparameters and guide rule pruning stopping criteria.
    \item \textbf{Test Partition ($\mathcal{D}_{\text{test}}$):} 20\% ($N_{\text{test}} = 154$; 101 non-diabetic, 53 diabetic). Strictly held out to evaluate final generalization fidelity and accuracy.
\end{enumerate}

\subsection{Leakage-Free Imputation}
For constrained clinical features (\texttt{Glucose}, \texttt{BloodPressure}, \texttt{SkinThickness}, \texttt{Insulin}, and \texttt{BMI}), zeros are mapped to \texttt{NaN} and imputed with the \textbf{training partition median}:
\begin{equation}
    x_{ij}^{\text{imputed}} = \begin{cases} 
    \text{Median}\left(\{ x_{lj} \mid x_{lj} > 0, l \in \mathcal{D}_{\text{train}} \}\right) & \text{if } x_{ij} = 0 \text{ and } j \in \mathcal{Z} \\
    x_{ij} & \text{otherwise}
    \end{cases}
\end{equation}
Calculated training medians are: \texttt{Glucose}: 117.0 mg/dL, \texttt{BloodPressure}: 72.0 mm Hg, \texttt{SkinThickness}: 29.0 mm, \texttt{Insulin}: 125.0 $\mu\text{U/mL}$, and \texttt{BMI}: 32.3 $\text{kg/m}^2$. Validation and test records are imputed using these exact training medians.

\section{Dual Feature Scaling: Standardization vs. Min-Max Normalization}

The neural network and the neuro-fuzzy surrogate require distinct scaling strategies:
\begin{itemize}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Standardization for Neural Networks:} Gradient descent requires zero-mean, unit-variance standardized inputs ($x_{ij}^{\text{std}} = (x_{ij} - \mu_j^{\text{train}}) / \sigma_j^{\text{train}}$) to ensure well-conditioned gradients.
    \item \textbf{Min-Max Normalization for ANFIS:} The fuzzy system requires inputs mapped to $[0, 1]$ ($\tilde{x}_{ij} = (x_{ij} - x_j^{\min}) / (x_j^{\max} - x_j^{\min})$) to ensure consistent Gaussian membership widths. For clinical presentation, membership parameters are mapped back to original medical units via exact inverse transformation:
    \begin{equation}
        c_j^{\text{orig}} = x_j^{\min} + c_j^{\text{scaled}} \cdot (x_j^{\max} - x_j^{\min}), \qquad \sigma_j^{\text{orig}} = \sigma_j^{\text{scaled}} \cdot (x_j^{\max} - x_j^{\min})
    \end{equation}
\end{itemize}

\section{Synthetic Oracle Neighborhood Sampling}

With only $N_{\text{train}} = 460$ real patients in an 8-dimensional space, training an explainer solely on empirical samples risks decision boundary underfitting. We implement a \textbf{Synthetic Oracle Sampling Engine} (\texttt{glassbox/oracle.py}) generating $N_{\text{synth}} = 2{,}000$ artificial profiles: (1)~\textbf{Gaussian Jitter (70\%):} $\tilde{z}_i = \text{clip}(\tilde{x}_i + \epsilon, 0, 1)$, $\epsilon \sim \mathcal{N}(0, 0.15^2 I)$; and (2)~\textbf{Uniform Exploration (30\%):} sampled across $[0, 1]^k$ to probe interstitial regions.

These synthetic patients do not possess ground-truth clinical labels. Instead, they are queried directly against the trained black-box neural network:
\begin{equation}
    t_{\text{synth}, i} = f_{\text{NN}}(z_i) = \text{blackbox.predict\_proba}(z_i) \in [0, 1]
\end{equation}
The augmented pool ($N_{\text{pool}} = 460 + 2{,}000 = \mathbf{2{,}460\text{ samples}}$) provides a dense representation of the black box's continuous risk landscape across the entire physiological input space.
"""
