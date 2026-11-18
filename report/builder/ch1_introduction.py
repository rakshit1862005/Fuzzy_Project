# Chapter 1: Introduction and Problem Definition (Aligned with EEO1003 and EE Dept Context)

CHAPTER_1 = r"""
\chapter{Introduction and Problem Definition}

\section{Context: Deep Learning in Healthcare and Clinical Diagnostics}

The integration of artificial intelligence (AI) and machine learning into clinical decision support systems (CDSS) represents a major advancement in chronic disease management, where early detection of type 2 diabetes mellitus prevents severe cardiovascular and renal complications. Modern diagnostic pipelines frequently employ feedforward artificial neural networks (ANNs), such as multi-layer perceptrons (MLPs) trained with backpropagation (Haykin, 1994; Fausett, 1993), to model non-linear physiological interactions across clinical features including plasma glucose, body mass index (BMI), age, maternal pregnancy count, and blood pressure.

However, clinical adoption of neural networks faces a fundamental hurdle: \textbf{opacity}. Because input measurements are transformed across layers of non-linear weights and activations, clinicians cannot trace why a patient was assigned an 84\% diabetes probability. Without verifiable decision criteria, physicians cannot safely rely on automated predictions for patient care.

\section{The Clinical Dilemma: High Accuracy vs. Dangerous Opacity}

Deploying neural networks in clinical environments introduces four major challenges: (1)~\textbf{Verification Necessity:} clinicians must confirm which physiological factors drove a diagnosis before prescribing treatment; (2)~\textbf{Confounding Artifacts:} black boxes can exploit dataset artifacts or missing-data patterns, hiding flaws behind high overall accuracy; (3)~\textbf{Patient Communication:} medical ethics requires explaining diagnostic risk in plain language rather than abstract weight matrices; and (4)~\textbf{Shortcomings of Local Attribution Tools:} popular local tools like LIME and SHAP output disconnected feature bars that fluctuate across runs and fail to provide auditable hospital-wide IF-THEN policies.

\section{Formal Problem Statement: Post-Hoc Rule Extraction}

Let a trained neural network be defined as a mapping $f_{\text{NN}}: \mathcal{X} \to \mathcal{Y}$, where $\mathcal{X} \subseteq \mathbb{R}^d$ represents the clinical input space ($d=8$ in PIDD) and $\mathcal{Y} \in [0, 1]$ represents the predicted probability of diabetes mellitus.

This project addresses the challenge of \textbf{Global Post-Hoc Rule Extraction}. Developed for the course \textit{Fuzzy Logic and Neural Networks (EEO1003)} offered by the Department of Electrical Engineering at Punjab Engineering College (Deemed to be University), Chandigarh, this project designs an interpretable neuro-fuzzy surrogate $f_{\text{fuzzy}}$ that extracts human-understandable IF-THEN rules from the trained neural network:
\begin{equation}
    \text{Rule } r: \quad \textbf{IF } \tilde{x}_1 \text{ is } A_{r1} \textbf{ AND } \dots \textbf{ AND } \tilde{x}_k \text{ is } A_{rk} \quad \textbf{THEN } \text{Risk } = p_r
\end{equation}
where $A_{rj}$ denotes an intuitive linguistic term (\textit{Low}, \textit{Medium}, or \textit{High}) defined by a Gaussian membership function, and $p_r \in [0, 1]$ is a crisp scalar risk consequence.

The surrogate must satisfy two clear engineering objectives:
\begin{enumerate}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{High Fidelity ($\mathcal{F}$):} The extracted fuzzy rules must accurately match the neural network's classifications across test patients: $\mathcal{F}(f_{\text{fuzzy}}, f_{\text{NN}}; \mathcal{D}_{\text{test}}) = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}[ \hat{y}^{\text{fuzzy}}_i = \hat{y}^{\text{NN}}_i ] \to 1.0$.
    \item \textbf{High Interpretability ($\mathcal{I}$):} The rule base must be compact enough for a human doctor to read in one glance: few rules ($R \le 5$), short conditions (average length $\le 3$), and stable linguistic partitions.
\end{enumerate}

\section{Regulatory Directives and the Glass-Box Philosophy}

Interpretable AI is increasingly mandated by international healthcare regulations: the \textbf{EU AI Act} classifies medical diagnostic tools as \textit{High-Risk}, requiring human clinical oversight; \textbf{GDPR} guarantees a patient's ``Right to Explanation''; and \textbf{U.S. FDA SaMD Guidelines} require auditable decision pathways. To satisfy these clinical and regulatory requirements, this project adopts an explainable glass-box philosophy: train a high-accuracy neural network first, query it as an oracle teacher across patient cases, and train a compact neuro-fuzzy inference system to explain the network's reasoning in plain medical language. This synthesizes three engineering pillars: (1)~\textbf{Pedagogical Rule Extraction} (Andrews et al., 1995); (2)~\textbf{Fuzzy Logic and Linguistic Variables} using smooth continuous shouldered Gaussian curves (Ross, 2010); and (3)~\textbf{Adaptive Neuro-Fuzzy Learning (ANFIS)} with hybrid optimization and semantic constraints (Jang, 1993).

\section{Project Objectives, Scope, and Report Structure}

\textbf{Core Objectives:} (1)~Train and cross-validate an 8-8-1 MLP on the benchmark PIDD cohort, applying Permutation Feature Importance to select the top 4 clinical drivers; (2)~Group patient data using Fuzzy C-Means clustering to initialize compact TSK rules; (3)~Apply hybrid learning under semantic ordering constraints ($c_{\text{Low}} < c_{\text{Med}} < c_{\text{High}}$); and (4)~Implement greedy backward pruning to achieve $>92\%$ fidelity with just 4 readable rules, closed-form arithmetic traceability, and live web telemetry.

\textbf{Scope and Academic Boundaries:} As fourth-year engineering students completing this course under the Department of Electrical Engineering, our focus is on practical engineering synthesis and implementation—integrating feature selection, clustering, hybrid learning, pruning, and reactive streaming telemetry—rather than claiming theoretical mathematical novelty over foundational literature. The system is evaluated on the benchmark PIDD cohort ($N=768$).

\textbf{Report Organization:} Chapter~2 reviews XAI taxonomies and fuzzy foundations; Chapter~3 details data preprocessing; Chapter~4 presents neural network training and feature selection; Chapter~5 derives the neuro-fuzzy rule extraction engine; Chapter~6 details software architecture and the web laboratory; Chapter~7 evaluates fidelity and complexity; Chapter~8 walks through clinical case studies; Chapter~9 discusses engineering requirements; and Chapter~10 presents conclusions.
"""
