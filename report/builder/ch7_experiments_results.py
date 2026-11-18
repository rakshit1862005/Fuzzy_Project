# Chapter 7: Experimental Evaluation, Results and Critical Analysis

CHAPTER_7 = r"""
\chapter{Experimental Evaluation, Results and Critical Analysis}

\section{Evaluation Framework and Performance Metrics}

Evaluating an explainable surrogate requires metrics that quantify both imitation fidelity and cognitive readability:
\begin{itemize}
    \setlength{\itemsep}{0pt}
    \setlength{\parskip}{0pt}
    \item \textbf{Classification Fidelity ($\mathcal{F}$):} Proportion of holdout test instances where the surrogate's discrete diagnostic prediction matches the black box: $\mathcal{F} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \mathbb{I}[\hat{Y}_{\text{fuzzy}}(x_i) = \hat{Y}_{\text{NN}}(x_i)]$.
    \item \textbf{Confident Fidelity ($\mathcal{F}_{\text{conf}}$):} Evaluated on confident cases where $P(\text{diabetic}) \notin [0.35, 0.65]$.
    \item \textbf{Boundary Fidelity ($\mathcal{F}_{\text{bound}}$):} Evaluated on ambiguous borderline cases where $P(\text{diabetic}) \in [0.35, 0.65]$.
    \item \textbf{Probability MAE ($\text{MAE}_{\text{prob}}$):} Average absolute continuous probability deviation: $\frac{1}{N_{\text{test}}} \sum |y(\tilde{x}_i) - t_i|$.
    \item \textbf{Interpretability Metrics:} Active rule count ($R$), total conditions ($\sum |A_r|$), average rule length ($\bar{L}$), and semantic order integrity percentage ($c_{\text{Low}} < c_{\text{Med}} < c_{\text{High}}$).
\end{itemize}

\section{Main Model Benchmark: Full vs. Pruned Rule Base}

Table~\ref{tab:main_model_results} summarizes empirical performance on the 154-patient holdout test partition under default settings ($k=4, C=8, \delta_{\max}=0.10, \Delta_{\text{fid}}=0.02, \Delta_{\text{mae}}=0.03$, seed 42).

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.75}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash\hsize=1.6\hsize}X >{\centering\arraybackslash\hsize=0.8\hsize}X >{\centering\arraybackslash\hsize=1.0\hsize}X >{\centering\arraybackslash\hsize=0.9\hsize}X >{\centering\arraybackslash\hsize=0.9\hsize}X >{\centering\arraybackslash\hsize=1.0\hsize}X >{\centering\arraybackslash\hsize=1.0\hsize}X >{\centering\arraybackslash\hsize=0.8\hsize}X}
\toprule
\textbf{Surrogate Stage} & \textbf{Rules ($R$)} & \textbf{Conditions} & \textbf{Avg Length} & \textbf{Fidelity} & \textbf{Boundary Fid} & \textbf{Confident Fid} & \textbf{Prob MAE} \\
\midrule
Full Model (Pre-Pruning) & 8 & 32 & 4.00 & \textbf{0.948} & 0.829 & \textbf{0.983} & 0.059 \\
\textbf{Simplified Model (Post-Pruning)} & \textbf{4} & \textbf{12} & \textbf{3.00} & \textbf{0.922} & 0.714 & \textbf{0.983} & 0.072 \\
\bottomrule
\end{tabularx}
\caption[Surrogate Performance Before and After Pruning]{Empirical test-set performance of the main model before and after backward greedy pruning.}
\label{tab:main_model_results}
\end{table}

Greedy backward pruning achieves a massive simplification: active conditions are reduced by \textbf{62.5\%} (from 32 down to 12) and rules are halved ($8 \to 4$). Crucially, on confident predictions ($P \notin [0.35, 0.65]$), the pruned model retains an exceptional \textbf{98.3\% fidelity}—matching the unpruned system identically. The aggregate 2.6\% fidelity difference is confined entirely to the ambiguous decision boundary corridor ($P \in [0.35, 0.65]$).

\section{The Extracted Clinical Rule Base and Linguistic Parameters}

Table~\ref{tab:final_rulebase} presents the complete 4-rule fuzzy rule base extracted by the framework.

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.75}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{c X c l c r}
\toprule
\textbf{Rule} & \textbf{Extracted Linguistic Antecedent (IF)} & \textbf{THEN Risk} & \textbf{Clinical Meaning} & \textbf{Decision} & \textbf{Coverage} \\
\midrule
\textbf{R1} & \textbf{IF} Glucose is Low ($\sim$102 mg/dL) \textbf{AND} BMI is Low ($\sim$29) \textbf{AND} Age is Low ($\sim$27) \textbf{AND} Pregnancies is Low ($\sim$1.5) & 0.033 & Very Low Risk & Non-diabetic & 44.6\% \\
\midrule
\textbf{R2} & \textbf{IF} Glucose is Medium ($\sim$133 mg/dL) \textbf{AND} Pregnancies is Medium ($\sim$4.4) & 0.352 & Low Risk & Non-diabetic & 33.3\% \\
\midrule
\textbf{R3} & \textbf{IF} Glucose is Medium ($\sim$133 mg/dL) \textbf{AND} BMI is High ($\sim$43) \textbf{AND} Age is Medium ($\sim$35) & 0.853 & Very High Risk & Diabetic & 5.0\% \\
\midrule
\textbf{R4} & \textbf{IF} Glucose is High ($\sim$155 mg/dL) \textbf{AND} BMI is Medium ($\sim$37) \textbf{AND} Age is Medium ($\sim$35) & 0.852 & Very High Risk & Diabetic & 17.2\% \\
\bottomrule
\end{tabularx}
\caption[Finalized 4-Rule Neuro-Fuzzy Rule Base]{The finalized 4-rule neuro-fuzzy rule base extracted from the black-box neural network.}
\label{tab:final_rulebase}
\end{table}

The extracted rules directly mirror endocrinological consensus: \textbf{Rule 1} identifies young, lean patients with low glycemic markers as virtually risk-free ($p_1 = 0.033$); \textbf{Rule 2} identifies multiparous patients with mild glucose elevation whose risk remains below threshold ($p_2 = 0.352$); \textbf{Rule 3} demonstrates that even with moderate glucose, severe adiposity (BMI High $\ge 43\text{ kg/m}^2$) triggers high risk ($p_3 = 0.853$); and \textbf{Rule 4} captures classic metabolic decompensation combining marked hyperglycemia (Glucose High $\ge 155$ mg/dL) and elevated BMI ($p_4 = 0.852$).

Optimized parameters in medical units converge to: \texttt{Glucose} (Low: $\le 102.3$, Med: $132.7 \pm 15.8$, High: $\ge 155.1$ mg/dL); \texttt{BMI} (Low: $\le 28.7$, Med: $37.0 \pm 6.5$, High: $\ge 43.0\text{ kg/m}^2$); \texttt{Age} (Low: $\le 26.9$, Med: $35.0 \pm 10.7$, High: $\ge 56.3$ yrs); and \texttt{Pregnancies} (Low: $\le 1.5$, Med: $4.4 \pm 3.7$, High: $\ge 10.4$).

\section{Multi-Seed Statistical Stability Analysis}

To confirm robustness across stochastic initializations, the full pipeline was executed across 5 independent random seeds ($s \in \{0, 1, 2, 3, 4\}$). Table~\ref{tab:multi_seed_results} presents the aggregate results (mean $\pm$ standard deviation).

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.75}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X c c c c}
\toprule
\textbf{Surrogate Stage} & \textbf{Rule Count ($R$)} & \textbf{Total Conditions} & \textbf{Test-Set Fidelity} & \textbf{Test Accuracy} \\
\midrule
Before Pruning & 6.6 $\pm$ 0.9 & 26.4 $\pm$ 3.6 & 0.951 $\pm$ 0.007 & 0.709 $\pm$ 0.012 \\
\textbf{After Pruning} & \textbf{5.0 $\pm$ 1.6} & \textbf{12.8 $\pm$ 5.0} & \textbf{0.948 $\pm$ 0.011} & \textbf{0.706 $\pm$ 0.014} \\
\bottomrule
\end{tabularx}
\caption[Multi-Seed Statistical Stability Evaluation]{Multi-seed evaluation across 5 independent runs demonstrating statistical stability.}
\label{tab:multi_seed_results}
\end{table}

Post-pruning fidelity remains exceptionally stable across seeds ($94.8\% \pm 1.1\%$), confirming that high fidelity is an inherent property of the methodology rather than an artifact of seed 42.

\section{Interpretability vs. Fidelity Trade-off: Pareto Comparison with Decision Trees}

We benchmark our neuro-fuzzy surrogate against the standard baseline in interpretable surrogate modeling: \textbf{Decision Tree Surrogates} (Craven \& Shavlik, 1996) trained on black-box predictions across increasing leaf limits ($C \in \{2, 4, 6, 8, 16\}$). Figure~\ref{fig:tradeoff_plot} and Table~\ref{tab:tradeoff_table} present the empirical comparison.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.82\textwidth,keepaspectratio]{tradeoff.png}
    \caption[Pareto Frontier: ANFIS vs. Decision Trees]{The Fidelity vs. Interpretability Pareto Frontier: comparing our pruned neuro-fuzzy surrogate against decision tree surrogates across varying complexity levels.}
    \label{fig:tradeoff_plot}
\end{figure}

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.75}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash\hsize=1.5\hsize}X >{\centering\arraybackslash\hsize=1.0\hsize}X >{\centering\arraybackslash\hsize=0.8\hsize}X >{\centering\arraybackslash\hsize=1.0\hsize}X >{\centering\arraybackslash\hsize=0.9\hsize}X >{\centering\arraybackslash\hsize=0.8\hsize}X}
\toprule
\textbf{Surrogate Architecture} & \textbf{Target Capacity} & \textbf{Mean Rules} & \textbf{Mean Conditions} & \textbf{Mean Fidelity} & \textbf{Fidelity Std} \\
\midrule
Decision Tree Baseline & 2 leaves & 2.0 & 2.0 & 0.883 & $\pm$ 0.000 \\
ANFIS Simplified (Proposed) & 2 clusters & 2.0 & 7.2 & \textbf{0.891} & $\pm$ 0.008 \\
\midrule
Decision Tree Baseline & 4 leaves & 4.0 & 8.0 & 0.877 & $\pm$ 0.000 \\
ANFIS Simplified (Proposed) & 4 clusters & 3.2 & 6.2 & \textbf{0.932} & $\pm$ 0.006 \\
\midrule
Decision Tree Baseline & 8 leaves & 8.0 & 25.0 & 0.909 & $\pm$ 0.000 \\
\textbf{ANFIS Simplified (Main)} & \textbf{8 clusters} & \textbf{5.0} & \textbf{12.8} & \textbf{0.948} & $\pm$ \textbf{0.011} \\
\midrule
Decision Tree Baseline & 16 leaves & 16.0 & 67.0 & 0.935 & $\pm$ 0.000 \\
ANFIS Simplified (Proposed) & 16 clusters & 6.6 & 14.2 & \textbf{0.940} & $\pm$ 0.008 \\
\bottomrule
\end{tabularx}
\caption[Fidelity and Complexity Trade-off Comparison]{Fidelity and complexity comparison between the neuro-fuzzy surrogate and decision trees.}
\label{tab:tradeoff_table}
\end{table}

Figure~\ref{fig:tradeoff_plot} illustrates the Pareto frontier of explanatory fidelity versus condition count. As visually demonstrated, decision trees require deep branching (up to 16 leaves and 67 condition checks) to reach 93.5\% fidelity. In contrast, our pruned neuro-fuzzy surrogate achieves \textbf{94.8\% fidelity with only 5 rules and 12.8 conditions}—achieving superior fidelity with an \textbf{81\% reduction in total conditions}. This efficiency stems from fuzzy logic's smooth membership surfaces, which capture continuous diagonal clinical boundaries without stair-step splits.

\section{Ablation Studies: Validating Core Components}

To quantify the impact of each architectural component, systematic ablation experiments were conducted across 5 random seeds (Table~\ref{tab:ablation_results}).

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.75}
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash\hsize=1.65\hsize}X >{\centering\arraybackslash\hsize=0.8\hsize}X >{\centering\arraybackslash\hsize=0.85\hsize}X >{\centering\arraybackslash\hsize=0.8\hsize}X >{\centering\arraybackslash\hsize=1.05\hsize}X >{\centering\arraybackslash\hsize=0.95\hsize}X >{\centering\arraybackslash\hsize=0.9\hsize}X}
\toprule
\textbf{Experimental Variant} & \textbf{Fidelity} & \textbf{Prob MAE} & \textbf{Rules ($R$)} & \textbf{Conditions} & \textbf{Max Drift} & \textbf{Order \%} \\
\midrule
\textbf{Full Proposed Method} & \textbf{0.948} & \textbf{0.063} & \textbf{5.0} & \textbf{12.8} & \textbf{0.100} & \textbf{100\%} \\
No Oracle Sampling (Real Only) & 0.935 & 0.061 & 5.0 & 11.0 & 0.100 & 100\% \\
No Interpretability Constraints & 0.942 & 0.058 & 4.2 & 12.4 & \textbf{0.495} & \textbf{60\%} \\
Trained on True Labels Direct & 0.882 & 0.121 & 4.4 & 11.2 & 0.080 & 100\% \\
First-Order TSK Consequents & 0.943 & 0.055 & 3.6 & 5.2 & 0.100 & 100\% \\
\bottomrule
\end{tabularx}
\caption[Systematic Ablation Analysis of Components]{Systematic ablation analysis evaluating the impact of core algorithmic components.}
\label{tab:ablation_results}
\end{table}

The ablation results confirm: (1)~\textbf{Oracle Sampling} boosts fidelity from 0.935 to 0.948 by populating boundary regions; (2)~\textbf{Semantic Constraints} prevent severe semantic collapse (without constraints, centers drift by 0.495 and 40\% of orderings invert); (3)~\textbf{Pedagogical Imitation} is essential (training directly on true labels $y \in \{0, 1\}$ drops fidelity to 88.2\%); and (4)~\textbf{Zero-Order TSK} provides scalar clarity ($p_r \in [0, 1]$) while matching first-order fidelity.
"""
