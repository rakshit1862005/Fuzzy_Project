# Chapter 2: Literature Review and Theoretical Foundations (Explainable, Aligned with EEO1003 Handout)

CHAPTER_2 = r"""
\chapter{Literature Review and Theoretical Foundations}

\section{The Spectrum of Model Explainability}

Explainable Artificial Intelligence (XAI) addresses the critical challenge of understanding how complex machine learning systems make decisions. In academic literature and engineering practice, interpretability is organized along two primary axes: \textbf{intrinsic versus post-hoc}, and \textbf{local versus global} (Molnar, 2022), as summarized in Table~\ref{tab:xai_taxonomy}.

\begin{table}[htbp]
    \centering
    \footnotesize
    \renewcommand{\arraystretch}{0.85}
    \begin{tabularx}{\textwidth}{l X X}
    \toprule
    \textbf{Dimension} & \textbf{Local Scope (Individual Prediction)} & \textbf{Global Scope (Entire Model Logic)} \\
    \midrule
    \textbf{Intrinsic} & Linear regression weights; single branch of a shallow decision tree. & Full shallow decision tree; Rule-based expert system (Mamdani \& Assilian, 1975). \\
    \midrule
    \textbf{Post-Hoc} & LIME (Ribeiro et al., 2016); Kernel SHAP (Lundberg \& Lee, 2017). & Decision Tree Surrogate (TREPAN); \textbf{Glass-Box Neuro-Fuzzy Surrogate (Proposed)}. \\
    \bottomrule
    \end{tabularx}
    \caption[Taxonomy of Explainable AI Methodologies]{Taxonomy of Explainable AI methodologies across scope and operational paradigm.}
    \label{tab:xai_taxonomy}
\end{table}

\textbf{Intrinsic models} (such as linear regression or small decision trees) are easy to understand because their structure is simple. However, medical conditions involve complicated, non-linear relationships that simple linear formulas cannot capture accurately. On the other hand, \textbf{post-hoc explainability} allows us to train a high-accuracy neural network first, and then build a transparent "surrogate" model that extracts human-understandable rules from it.

\section{Local Perturbation Methods and Clinical Limitations}

Currently, the most common explainability techniques in industry are local perturbation tools:

\subsection{Local Interpretable Model-agnostic Explanations (LIME)}
LIME (Ribeiro et al., 2016) explains a single prediction by creating slight random variations around that patient, checking what the neural network predicts for each variation, and fitting a simple linear line locally. While conceptually straightforward, LIME's random sampling produces slightly different explanation weights every time it runs on the exact same patient, which undermines clinical confidence.

\subsection{SHapley Additive exPlanations (SHAP)}
SHAP (Lundberg \& Lee, 2017) applies cooperative game theory to assign an importance credit (Shapley value) to each feature. While mathematically principled, SHAP assumes features act independently in an additive sum ($\sum \phi_j$). This fails to convey medical interactions---for instance, high glucose combined with high BMI creates compounded diabetes risk that an additive sum cannot clearly represent.

\subsection{Clinical Deficiencies of Local Perturbations}
Local tools exhibit critical shortcomings in hospital workflows: (1)~\textbf{No Global Policy:} They output isolated bar charts for one patient at a time, but do not provide a set of rules that doctors can audit as hospital policy; (2)~\textbf{Sampling Artifacts:} Random perturbations can generate medically impossible patient records (such as zero blood pressure combined with high glucose); and (3)~\textbf{Lack of Plain-Language Rules:} Bar charts do not state clear IF-THEN diagnostic conditions.

\section{Global Rule Extraction Taxonomies}

To provide complete model auditability, researchers study \textbf{Global Rule Extraction}. As surveyed by Andrews, Diederich, and Tickle (1995), extraction methods fall into three broad classes:
\begin{enumerate}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Decompositional:} Inspects individual neurons and weight matrices; heavily dependent on network topology and activation functions.
    \item \textbf{Pedagogical (Black-Box Imitation):} Treats the neural network strictly as an oracle teacher, learning transparent IF-THEN rules directly from network inputs and outputs.
    \item \textbf{Eclectic:} Combines black-box sampling with internal weight analysis.
\end{enumerate}
Our \textbf{Glass-Box Insight} project follows the \textbf{pedagogical approach}. By treating the trained neural network as an oracle, the fuzzy extraction pipeline is modular and can explain any neural classifier regardless of internal architecture.

\section{Symbolic Surrogates: Decision Trees (TREPAN) and Crisp Limits}

Early surrogate systems used decision trees (TREPAN; Craven \& Shavlik, 1996). However, traditional "crisp" decision trees suffer from rigid boundary thresholds. For example, if a split threshold is set at $\text{Glucose} \le 126.0\text{ mg/dL}$, a patient with 125.9 mg/dL is categorized as completely healthy while 126.1 mg/dL is categorized as diabetic. This abrupt jump fails to reflect the continuous nature of human biology.

\section{Fuzzy Set Theory and Linguistic Modeling}

To handle gradual transitions naturally, Lotfi A. Zadeh (1965) introduced \textbf{Fuzzy Set Theory}, a cornerstone of modern intelligent systems engineering (Ross, 2010; Sivanandam et al., 2007). In classical set theory, an element either belongs to a set or does not ($\chi_A(x) \in \{0, 1\}$). In fuzzy set theory, membership is continuous:
\begin{equation}
    \mu_A(x) \in [0, 1]
\end{equation}
This allows us to model \textbf{linguistic variables} (Zadeh, 1975) like \textit{Glucose} using medical terms: \textit{Low}, \textit{Medium}, and \textit{High}.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.80\textwidth,keepaspectratio]{crisp_vs_fuzzy.png}
    \caption[Crisp vs. Fuzzy Set Membership]{Conceptual comparison between crisp binary set membership and continuous fuzzy set membership across the clinical glycemic continuum.}
    \label{fig:crisp_vs_fuzzy}
\end{figure}

Figure~\ref{fig:crisp_vs_fuzzy} illustrates the fundamental contrast between crisp and fuzzy representations of blood glucose. In the crisp model (left), a rigid cutoff at 126 mg/dL forces an unnatural step-change: 125.9 mg/dL is categorized as non-diabetic ($\mu=0$), while 126.1 mg/dL is classified as diabetic ($\mu=1$). In contrast, the continuous fuzzy membership model (right) provides smooth, overlapping transitions that reflect genuine clinical gradualness, recognizing borderline states without arbitrary cliffs.

\subsection{Shouldered Gaussian Memberships}
Standard bell-shaped Gaussians decay toward zero at extreme values. To ensure that severely ill patients (e.g., blood glucose $\ge 200$ mg/dL) do not receive zero membership in \textit{High}, we implement \textbf{monotonic flat boundary shoulders}: \textit{Low} remains flat at 1.0 for all values below its center, and \textit{High} remains flat at 1.0 for all values above its center.

\section{Fuzzy Inference Systems: Zero-Order Takagi-Sugeno-Kang}

In fuzzy systems engineering (Ross, 2010; Sivanandam et al., 2007), a Fuzzy Inference System (FIS) maps input linguistic terms to outputs. While Mamdani systems use fuzzy sets for outputs, the \textbf{zero-order Takagi-Sugeno-Kang (TSK)} model (Takagi \& Sugeno, 1985) uses crisp numeric risk values:
\begin{equation}
    \text{Rule } r: \quad \textbf{IF } x_1 \text{ is } A_{r1} \textbf{ AND } \dots \textbf{ AND } x_k \text{ is } A_{rk} \quad \textbf{THEN } \text{Risk } = p_r \in [0, 1]
\end{equation}
The final output is computed by taking the weighted average of all rule outputs:
\begin{equation}
    y(x) = \sum_{r=1}^R \bar{w}_r(x) \cdot p_r
\end{equation}
where $\bar{w}_r$ is the normalized firing strength of rule $r$. This provides exact closed-form arithmetic traceability that anyone can verify with a handheld calculator.

\section{Neuro-Fuzzy Systems: ANFIS and Hybrid Learning}

An Adaptive Neuro-Fuzzy Inference System (ANFIS) (Jang, 1993; Sivanandam et al., 2007) connects neural networks and fuzzy logic into a 5-layer network:
\begin{itemize}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Layer 1:} Calculates membership values $\mu(x)$ for each feature in \textit{Low}, \textit{Medium}, \textit{High}.
    \item \textbf{Layer 2:} Calculates rule firing strengths using the product T-norm: $w_r = \mu_1 \times \mu_2 \dots$
    \item \textbf{Layer 3:} Normalizes firing strengths: $\bar{w}_r = w_r / \sum w$.
    \item \textbf{Layer 4 \& 5:} Multiplies by consequent risk scores $p_r$ and sums them to obtain final risk $y(x)$.
\end{itemize}
ANFIS trains these parameters using \textbf{hybrid learning}: linear risk consequents $p_r$ are solved in one step via regularized least squares, while the membership curve centers and widths are fine-tuned using standard backpropagation gradient descent (Haykin, 1994; Fausett, 1993).

\section{Fuzzy C-Means Clustering to Mitigate Rule Explosion}

If we created every possible combination of 4 features with 3 linguistic terms, we would get $3^4 = \mathbf{81\text{ rules}}$ (and $3^8 = 6{,}561$ for 8 features), which is impossible for a doctor to read. To prevent rule explosion, we apply \textbf{Fuzzy C-Means (FCM)} clustering (Bezdek, 1981) to find 8 natural patient prototypes from the training data, initializing an initial rule base of just 8 compact rules.

\section{Fuzzy Interpretability Criteria and Research Gap}

Casillas et al. (2003) showed that a trustworthy fuzzy system requires both \textbf{low syntactic complexity} (few rules and conditions) and \textbf{semantic integrity} (linguistic terms must remain ordered: $c_{\text{Low}} < c_{\text{Med}} < c_{\text{High}}$). Table~\ref{tab:literature_comparison} compares existing methods with our proposed system.

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.80}
\setlength{\tabcolsep}{2.5pt}
\begin{tabularx}{\textwidth}{>{\bfseries\raggedright\arraybackslash}p{2.0cm} >{\raggedright\arraybackslash}p{1.9cm} >{\raggedright\arraybackslash}p{1.4cm} >{\raggedright\arraybackslash}p{2.1cm} >{\raggedright\arraybackslash}X}
\toprule
\textbf{Method} & \textbf{Explanation} & \textbf{Stability} & \textbf{Traceability} & \textbf{Clinical Auditability \& Readability} \\
\midrule
LIME (2016) & Local Attribution & Low & Local linear model & Bar chart; lacks unified global rule policy. \\
\midrule
SHAP (2017) & Local Shapley & High & Additive ($\sum \phi_j$) & Force plots; assumes feature independence. \\
\midrule
TREPAN (1996) & Global Decision Tree & Moderate & Crisp Boolean path & Tree diagram; discontinuous at thresholds. \\
\midrule
Standard ANFIS & Global Neuro-Fuzzy & High & Exact fuzzy rules & Suffers from rule explosion ($3^8 = 6{,}561$ rules). \\
\midrule
Glass-Box Insight & Global Neuro-Fuzzy & High & Closed-form algebraic trace & 4 readable rules; 12 conditions; semantic bounds; weak-match alert. \\
\bottomrule
\end{tabularx}
\caption[Comparative Evaluation of Explainable AI Frameworks]{Comparative evaluation of Explainable AI frameworks for clinical decision support.}
\label{tab:literature_comparison}
\end{table}

\textbf{Identified Research Gap:} Existing techniques force clinicians to choose between unstable local feature bars (LIME/SHAP), brittle crisp decision trees (TREPAN), or bloated fuzzy rule bases. Glass-Box Insight bridges this gap by combining feature selection, clustering, hybrid neuro-fuzzy learning, and rule pruning into a compact 4-rule glass-box model.
"""
