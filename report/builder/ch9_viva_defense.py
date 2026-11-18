# Chapter 9: Discussion, System Evaluation and Limitations

CHAPTER_9 = r"""
\chapter{Discussion, System Evaluation and Limitations}

\section{Evaluation Against Original Engineering Requirements}

To rigorously assess whether the Glass-Box Insight project fulfilled its original design specifications, Table~\ref{tab:traceability_table} maps each initial engineering challenge to its implemented computational mechanism, empirical evidence, and remaining limitations.

\begin{table}[htbp]
\centering
\small
\renewcommand{\arraystretch}{0.9}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash\hsize=0.75\hsize}X >{\raggedright\arraybackslash\hsize=0.85\hsize}X >{\raggedright\arraybackslash\hsize=1.05\hsize}X >{\raggedright\arraybackslash\hsize=1.35\hsize}X}
\toprule
\textbf{Challenge} & \textbf{Proposed Solution} & \textbf{Empirical Evidence} & \textbf{Remaining Limitations} \\
\midrule
\textbf{Neural Network Opacity} & Global pedagogical neuro-fuzzy surrogate. & 4 readable linguistic rules; 92.2\% test fidelity; per-patient decomposition. & Explains the surrogate; 7.8\% of patients experience surrogate divergence. \\
\midrule
\textbf{Curse of Dimensionality} & Permutation importance feature selection ($k=4$). & Saliency filter captures 82.9\% attribution; reduces grid rules from 6,561 to 8. & Ignores secondary interactions in excluded 4 features (BP, Insulin, Skin, DPF). \\
\midrule
\textbf{Linguistic Semantic Drift} & Order, drift bound, and width floor constraints. & 100\% monotonic ordering maintained; drift restricted to $\le 0.10$. & Fixed width floors can occasionally create wider-than-ideal term overlap. \\
\midrule
\textbf{Combinatorial Rule Explosion} & Fuzzy C-Means clustering + greedy backward pruning. & Reduces 32 conditions down to 12; maintains 98.3\% confident fidelity. & Greedy pruning explores single-step deletions; non-convex search space. \\
\midrule
\textbf{Decision Boundary Sparsity} & Synthetic Oracle Sampling (2,000 instances). & Prevents surrogate boundary collapse; boosts fidelity from 93.5\% to 94.8\%. & Synthetic samples generated via Gaussian perturbation rather than GANs. \\
\midrule
\textbf{Silent Failure on Outliers} & Match-Strength diagnostic ($\mathcal{S}_{\text{match}} < 0.20$). & Correctly alerts clinicians on atypical patient profiles (Case Study 5). & Requires manual clinical intervention when alert triggers. \\
\bottomrule
\end{tabularx}
\caption[System Traceability and Requirement Evaluation Matrix]{System traceability and requirement fulfillment evaluation matrix.}
\label{tab:traceability_table}
\end{table}

\section{Mathematical and Architectural Justifications}

Every structural design choice in the Glass-Box Insight framework was guided by formal mathematical principles:
\begin{enumerate}
    \setlength{\itemsep}{1pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Why Product T-Norm for AND Conjunctions?} We selected the algebraic product ($w = \prod \mu$) over the Gödel minimum ($w = \min \mu$). The minimum operator has zero gradient with respect to non-minimal inputs, preventing backpropagation through non-critical features. The product operator is smooth, infinitely differentiable, and sensitive to all antecedent conditions simultaneously.
    \item \textbf{Why Shouldered Gaussians?} Unshouldered Gaussians decay to zero at infinity, causing patients with extreme clinical measurements to have zero membership in all terms, triggering numerical underflow in Layer 3. Monotonic shoulders ensure infinite domain coverage at physiological extremes.
    \item \textbf{Why Ridge Regularization on Consequents?} In regions of the feature space where few training points reside, $W^T W$ can become ill-conditioned. Adding $\lambda_{\text{ridge}} I$ guarantees positive definiteness and prevents unbounded consequent weights.
\end{enumerate}

\section{Limitations and Threats to Validity}

A rigorous engineering evaluation must acknowledge intrinsic limitations:
\begin{enumerate}
    \setlength{\itemsep}{1pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Demographic and Geographic Specificity:} The PIDD dataset consists exclusively of adult females of Pima Indian heritage. Because diabetes pathophysiology varies across ethnic and demographic groups, the extracted rules cannot be deployed clinically across diverse global populations without multi-center prospective retraining.
    \item \textbf{Sample Size Constraints:} With $N = 768$ total patients, the holdout test set contains 154 individuals. While statistical significance is confirmed across 5 random seeds, validation on contemporary cohorts ($N > 10{,}000$) is required for clinical translation.
    \item \textbf{Information Exclusion in Feature Selection:} By restricting the surrogate to the top 4 features, the surrogate cannot account for subtle secondary interactions involving Blood Pressure, Serum Insulin, or Pedigree.
    \item \textbf{Term Overlap Artifacts:} As observed in Patient \#152, the minimum width constraint ($\sigma \ge 0.05$) occasionally results in wide membership overlap, causing a patient with BMI = 37 to exhibit 0.52 membership in Low and 0.53 in High simultaneously.
\end{enumerate}

\section{Future Research and Clinical Deployment Roadmap}

Future extensions of the Glass-Box Insight architecture should explore:
\begin{enumerate}
    \setlength{\itemsep}{1pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Interval Type-2 Fuzzy Sets:} Transitioning from Type-1 to Type-2 fuzzy sets to explicitly model intra-observer clinical measurement noise and laboratory assay variability.
    \item \textbf{Dynamic Temporal Surrogates:} Extending neuro-fuzzy extraction to recurrent neural networks (LSTMs) and Transformers for longitudinal electronic health records (EHR).
    \item \textbf{FHIR Standards Integration:} Packaging the surrogate and web laboratory as an HL7/FHIR (Fast Healthcare Interoperability Resources) compliant microservice, enabling direct plug-and-play integration into hospital Electronic Medical Record systems (e.g., Epic, Cerner).
\end{enumerate}
"""
