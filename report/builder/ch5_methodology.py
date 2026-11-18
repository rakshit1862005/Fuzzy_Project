# Chapter 5: Proposed Glass-Box Methodology: Neuro-Fuzzy Rule Extraction Engine (Explainable, Aligned with EEO1003)

CHAPTER_5 = r"""
\chapter{Proposed Methodology: Neuro-Fuzzy Rule Extraction Engine}

\section{End-to-End Architectural Pipeline}

The \textbf{Glass-Box Insight} framework executes a six-stage engineering pipeline that transforms opaque neural network predictions into a compact, semantically preserved, and mathematically traceable neuro-fuzzy rule base (Table~\ref{tab:methodology_pipeline}).

\begin{table}[htbp]
    \centering
    \footnotesize
    \renewcommand{\arraystretch}{0.80}
    \begin{tabularx}{\textwidth}{>{\bfseries\raggedright\arraybackslash}p{3.5cm} >{\raggedright\arraybackslash}X}
    \toprule
    \textbf{Pipeline Stage} & \textbf{Core Computational Process and Objectives} \\
    \midrule
    Stage 1: Salient Feature Ingestion & Ingest top $k=4$ features from Permutation Importance; scale inputs to $[0, 1]$ via min-max normalization. \\
    \midrule
    Stage 2: FCM Clustering & Group training patients ($N=2{,}460$) into $C=8$ multi-dimensional prototypes; map centroids to linguistic terms. \\
    \midrule
    Stage 3: Premise Formulation & Construct Gaussian membership functions with monotonic boundary shoulders to guarantee full clinical coverage. \\
    \midrule
    Stage 4: Hybrid Learning & Forward pass solves zero-order TSK consequent risk scalars via Ridge RLS; backward pass tunes premise terms via Adam. \\
    \midrule
    Stage 5: Semantic Constraints & Enforce natural linguistic ordering ($c_{\text{Low}} < c_{\text{Med}} < c_{\text{High}}$), bounded drift ($\le 0.10$), and minimum curve widths. \\
    \midrule
    Stage 6: Greedy Pruning & Systematically eliminate irrelevant conditions and redundant rules under validation fidelity and error checks ($8 \to 4$ rules). \\
    \bottomrule
    \end{tabularx}
    \caption[Six-Stage Glass-Box Extraction Architecture]{Sequential six-stage architecture of the Glass-Box Insight neuro-fuzzy rule extraction pipeline.}
    \label{tab:methodology_pipeline}
\end{table}

\section{Data-Driven Rule Generation via Fuzzy C-Means Clustering}

Rather than creating an unreadable grid of $3^4 = 81$ rules, initial rules are discovered directly from patient data using \textbf{Fuzzy C-Means (FCM)} clustering (Bezdek, 1981; Ross, 2010).

Let $\tilde{X} = \{\tilde{x}_1, \dots, \tilde{x}_N\} \subset [0, 1]^k$ represent the salient patient pool ($N = 2{,}460$, combining 460 real training patients and 2,000 synthetic neighborhood samples). FCM groups these patients into $C = 8$ clusters by minimizing the objective:
\begin{equation}
    J_m(U, V) = \sum_{i=1}^N \sum_{c=1}^C u_{ic}^m \|\tilde{x}_i - v_c\|^2, \quad \text{subject to } \sum_{c=1}^C u_{ic} = 1 \quad \forall i
\end{equation}
where $u_{ic} \in [0, 1]$ represents the fuzzy membership of patient $i$ in cluster $c$, $v_c \in [0, 1]^k$ is the cluster centroid vector, and $m = 2.0$ is the fuzziness exponent.

Each centroid $v_c$ represents a prototypical patient profile. To convert continuous centroid coordinates into discrete linguistic rules, each coordinate $v_{cj}$ is matched to the nearest linguistic term index $t \in \{0, 1, 2\}$:
\begin{equation}
    A_{rj} = \arg\min_{t \in \{0, 1, 2\}} |v_{rj} - c_{jt}^{(0)}|
\end{equation}
where $t=0$ corresponds to \textit{Low}, $t=1$ to \textit{Medium}, and $t=2$ to \textit{High}. Any duplicate rules produced by this mapping are merged, yielding an initial rule base of $R \le 8$ transparent IF-THEN rules.

\section{Linguistic Formulation and Shouldered Gaussian Memberships}

For each of the $k=4$ salient clinical features, the normalized range $[0, 1]$ is partitioned into 3 linguistic terms: \textit{Low} ($t=0$), \textit{Medium} ($t=1$), and \textit{High} ($t=2$), initialized at anchors $c_{j0}^{(0)} = 0.20, c_{j1}^{(0)} = 0.50, c_{j2}^{(0)} = 0.80$ with initial widths $\sigma_{jt}^{(0)} = 0.18$.

Standard Gaussian bell curves decay toward zero at extreme values, which causes severe clinical misclassification when a patient exhibits extreme hyperglycemia or high BMI. To prevent this, we implement \textbf{monotonic flat boundary shoulders}:
\begin{align}
    \mu_{j, \text{Low}}(\tilde{x}_j) &= \begin{cases} 1.0 & \text{if } \tilde{x}_j \le c_{j, \text{Low}} \\ \exp\left( -\frac{1}{2} \left( \frac{\tilde{x}_j - c_{j, \text{Low}}}{\sigma_{j, \text{Low}}} \right)^2 \right) & \text{if } \tilde{x}_j > c_{j, \text{Low}} \end{cases} \\
    \mu_{j, \text{Med}}(\tilde{x}_j) &= \exp\left( -\frac{1}{2} \left( \frac{\tilde{x}_j - c_{j, \text{Med}}}{\sigma_{j, \text{Med}}} \right)^2 \right) \\
    \mu_{j, \text{High}}(\tilde{x}_j) &= \begin{cases} \exp\left( -\frac{1}{2} \left( \frac{\tilde{x}_j - c_{j, \text{High}}}{\sigma_{j, \text{High}}} \right)^2 \right) & \text{if } \tilde{x}_j < c_{j, \text{High}} \\ 1.0 & \text{if } \tilde{x}_j \ge c_{j, \text{High}} \end{cases}
\end{align}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.76\textwidth,keepaspectratio]{membership_functions.png}
    \caption[Optimized Shouldered Gaussian Membership Functions]{Optimized shouldered Gaussian membership functions for Glucose, BMI, Age, and Pregnancies. Solid curves denote final trained parameters; dashed curves denote initial anchor positions.}
    \label{fig:membership_functions_plot}
\end{figure}

Figure~\ref{fig:membership_functions_plot} depicts the optimized membership topologies across all four salient clinical attributes. In each subplot, the initial quantile anchors (dashed lines) adapted smoothly during hybrid learning into the final trained membership curves (solid lines). Monotonic flat shoulders on the outer boundaries guarantee that severe clinical presentations (such as extreme hyperglycemia $\ge 180$ mg/dL or morbid obesity $\ge 45\text{ kg/m}^2$) retain 100\% membership in \textit{High} without tail decay. Crucially, semantic preservation constraints prevent overlapping collapse, ensuring distinct, medically intuitive linguistic partitions.

\section{Layer-by-Layer ANFIS System Dynamics}

The neuro-fuzzy surrogate is implemented as a 5-layer feedforward zero-order Takagi-Sugeno-Kang network:
\begin{itemize}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Layer 1 (Fuzzification):} Ingests normalized input $\tilde{x}_j$ and computes membership grades $\mu_{jt}(\tilde{x}_j) \in [0, 1]$ using Gaussian curve parameters (center $c_{jt}$ and width $\sigma_{jt}$).
    \item \textbf{Layer 2 (Rule Firing via Product T-Norm):} Multiplies membership grades across active antecedents:
    \begin{equation}
        w_r(\tilde{x}) = \prod_{j \in \mathcal{A}(r)} \mu_{j, A_{rj}}(\tilde{x}_j)
    \end{equation}
    where $\mathcal{A}(r)$ denotes the active antecedent features. If a condition is pruned ($A_{rj} = -1$), its multiplier is set to 1.0 (\textit{Don't Care}).
    \item \textbf{Layer 3 (Normalization):} Normalizes firing strengths to enforce a partition of unity:
    \begin{equation}
        \bar{w}_r(\tilde{x}) = \frac{w_r(\tilde{x})}{\sum_{s=1}^R w_s(\tilde{x}) + \epsilon}, \quad \text{such that } \sum_{r=1}^R \bar{w}_r(\tilde{x}) = 1.0
    \end{equation}
    \item \textbf{Layers 4 and 5 (Consequent Weighting \& Defuzzification):} In zero-order TSK, each rule has a scalar risk score $p_r \in [0, 1]$. Layer~4 computes the weighted risk $O_{4, r} = \bar{w}_r(\tilde{x}) p_r$, and Layer~5 sums them to produce the final predicted probability:
    \begin{equation}
        y(\tilde{x}) = \sum_{r=1}^R \bar{w}_r(\tilde{x}) \cdot p_r \in [0, 1]
    \end{equation}
\end{itemize}

\section{Hybrid Learning Optimization Algorithm}

To train the ANFIS surrogate efficiently, Jang's \textbf{hybrid learning algorithm} (Jang, 1993) splits the optimization into two complementary phases:

\subsection{Forward Step: Consequent Parameter Optimization via Ridge RLS}
When the premise membership curves are held fixed, the output equation $y(\tilde{x}) = \sum \bar{w}_r p_r$ is purely linear with respect to the rule risk consequences $p \in \mathbb{R}^R$. Let $W \in \mathbb{R}^{N \times R}$ be the matrix of normalized firing strengths and $t \in [0, 1]^N$ be the target probabilities from the neural network. We solve for the optimal consequent vector $p^*$ in a single closed-form step using regularized ridge least squares:
\begin{equation}
    p^* = \left( W^T W + \lambda_{\text{ridge}} I_R \right)^{-1} W^T t, \qquad p_r \leftarrow \text{clip}(p_r^*, 0, 1)
\end{equation}
where $\lambda_{\text{ridge}} = 10^{-4}$ guarantees numerical stability and prevents matrix singularity.

\subsection{Backward Step: Premise Optimization via Constrained Adam}
With consequents $p$ held fixed, the membership center and width parameters $\theta = \{c_{jt}, \sigma_{jt}\}$ are updated to minimize the mean squared error loss $\mathcal{L} = \frac{1}{N} \sum (y_i - t_i)^2$. Gradients are backpropagated through the chain rule, and parameters are updated using the Adam optimizer ($\eta = 0.01$).

\section{Semantic Integrity Preservation Constraints}

Without constraints, gradient descent can cause \textbf{semantic collapse} (e.g., the \textit{Low} center drifting higher than the \textit{High} center, or curves shrinking to needle-thin delta functions). To keep the rules medically meaningful, three constraints are enforced after each training step:
\begin{enumerate}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Strict Linguistic Ordering:} $c_{j, \text{Low}} + 0.08 \le c_{j, \text{Med}} \le c_{j, \text{High}} - 0.08$. This guarantees that \textit{Low} is always less than \textit{Medium}, and \textit{Medium} is always less than \textit{High}.
    \item \textbf{Bounded Anchor Drift:} Centers cannot drift more than $\pm 0.10$ from their original initial data anchors ($|c_{jt} - c_{jt}^{(0)}| \le 0.10$).
    \item \textbf{Minimum Curve Width Floor:} Widths must satisfy $\sigma_{jt} \ge 0.05$ to prevent sharp discontinuous spikes.
\end{enumerate}

\section{Two-Check Backward Greedy Pruning Algorithm}

The initial rule base contains 8 rules with $8 \times 4 = 32$ conditions. We implement a \textbf{Two-Check Backward Greedy Pruning Algorithm} (\texttt{glassbox/pruning.py}) that simplifies the rule base in two stages:
\begin{enumerate}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Condition Pruning:} Iteratively removes conditions from rules (setting $A_{rj} \leftarrow -1$).
    \item \textbf{Rule Pruning:} Eliminates entire redundant or inactive rules.
\end{enumerate}
At each candidate deletion, consequent risk scores $p$ are re-solved in under 5 ms via Ridge RLS, and the deletion is permanently accepted \textbf{if and only if} it passes both validation tests:
\begin{align}
    \textbf{Check 1 (Fidelity Drop Check):} & \quad \mathcal{F}_{\text{new}}(\mathcal{D}_{\text{val}}) \ge \mathcal{F}_{\text{baseline}}(\mathcal{D}_{\text{val}}) - 0.02 \\
    \textbf{Check 2 (Probability Error Check):} & \quad \text{MAE}_{\text{new}}(\mathcal{D}_{\text{val}}) \le \text{MAE}_{\text{baseline}}(\mathcal{D}_{\text{val}}) + 0.03
\end{align}
Under random seed 42, pruning removes 14 redundant conditions and merges duplicate rules, reducing the system from \textbf{8 rules and 32 conditions} down to \textbf{4 readable rules and 12 conditions} while preserving over 92\% fidelity.

\section{Decision Traceability and Weak-Match Diagnostics}

Unlike black-box models, zero-order TSK models offer \textbf{100\% closed-form mathematical traceability}. Any patient risk score is algebraically decomposed into individual rule contributions:
\begin{equation}
    y(x) = \sum_{r=1}^R \bar{w}_r(x) \cdot p_r, \qquad \text{where } \text{Share}_r(x) = \frac{\bar{w}_r(x) \cdot p_r}{y(x)} \times 100\%
\end{equation}

To prevent silent automated errors on atypical patients, we introduce the \textbf{Match Strength Metric}: $\mathcal{S}_{\text{match}}(x) = \max_{r} w_r(x) \in [0, 1]$. If $\mathcal{S}_{\text{match}} < 0.20$, the system triggers an active \textbf{Weak-Match Diagnostic Warning}, alerting clinicians that the patient's physiological profile falls outside the active rule prototypes, mandating direct human clinical evaluation.
"""
