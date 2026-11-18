# Chapter 8: Comprehensive Clinical Case Studies and Arithmetic Trace

CHAPTER_8 = r"""
\chapter{Comprehensive Clinical Case Studies and Arithmetic Trace}

To demonstrate how Glass-Box Insight operates in clinical practice, this chapter presents representative case studies from the holdout test partition ($N_{\text{test}} = 154$), verified against the main trained model (seed 42).

\section{Case Study 1: Confident Diabetic Classification (Patient \#152)}

Patient \#152 (data row 425) presents classic metabolic syndrome: 31-year-old female, 4 pregnancies, Glucose = 184 mg/dL, Blood Pressure = 78 mm Hg, Skinfold = 39 mm, Serum Insulin = 277 $\mu\text{U/mL}$, BMI = 37.0 $\text{kg/m}^2$, Pedigree = 0.264. True outcome: \textbf{Diabetic (1)}.

\subsection{Black-Box Neural Network vs. Neuro-Fuzzy Arithmetic Walkthrough}
The black box ingests standardized inputs $x^{\text{std}}$, activating hidden units $h = [0, 1.64, 0, 1.59, 0, 0, 0.41, 0]^T$ to emit logit $z^{[2]} = 1.634$ and risk probability $\hat{y}_{\text{NN}} = \sigma(1.634) = \mathbf{0.837}$ (\textbf{Diabetic, 84\% risk}).

The neuro-fuzzy explainer scales salient inputs to $[0, 1]$: $\tilde{x} = [\text{Glu} = 0.908, \text{BMI} = 0.384, \text{Age} = 0.167, \text{Preg} = 0.286]$. Layer~1 evaluates memberships: $\mu_{\text{Glu}} = [0.019, 0.005, \mathbf{1.000}]$, $\mu_{\text{BMI}} = [0.519, \mathbf{1.000}, 0.532]$, $\mu_{\text{Age}} = [0.946, 0.933, 0.000]$, $\mu_{\text{Preg}} = [0.800, 0.994, 0.008]$. Layer~2 evaluates product T-norms:
\begin{equation}
    w_1 = 0.0074, \quad w_2 = 0.0052, \quad w_3 = 0.0026, \quad w_4 = 1.000 \times 1.000 \times 0.933 = \mathbf{0.9327}
\end{equation}
Rule 4 fires with overwhelming dominance ($w_4 = 0.9327$). Normalizing in Layer~3 yields $\bar{w} = [0.008, 0.005, 0.003, \mathbf{0.984}]$. Defuzzification in Layer~5 produces:
\begin{align}
    y(\tilde{x}) &= 0.008(0.033) + 0.005(0.352) + 0.003(0.853) + 0.984(0.852) \\
    &= \mathbf{0.843} \implies \text{Diabetic (84\% Risk)}
\end{align}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth,keepaspectratio]{case_confident_diabetic.png}
    \caption[Diagnostic Explanation for Confident Diabetic Case]{Diagnostic explanation summary for Confident Diabetic Patient \#152, illustrating closed-form rule decomposition.}
    \label{fig:case_confident_diabetic}
\end{figure}

Figure~\ref{fig:case_confident_diabetic} illustrates the diagnostic telemetry generated for Patient \#152. Both models achieve perfect consensus at 84\% diabetic probability. The bar chart demonstrates that \textbf{Rule 4} accounts for \textbf{98.4\%} of the explanation: \textit{``IF Glucose is High (184 mg/dL) AND BMI is Medium (37.0 $\text{kg/m}^2$) AND Age is Medium (31 yrs) THEN Risk = 0.85.''} With match strength $\mathcal{S}_{\text{match}} = 0.93$, the explanation is certified as highly reliable.

\section{Case Study 2: Confident Healthy Classification (Patient \#29)}

Patient \#29 represents an uncompromised young individual: Glucose = 89 mg/dL, BMI = 28.1 $\text{kg/m}^2$, Age = 21 yrs, Pregnancies = 1.0. True Outcome: \textbf{Non-diabetic (0)}. The black box predicts $\hat{y}_{\text{NN}} = \mathbf{0.02}$; the neuro-fuzzy surrogate predicts $y = \mathbf{0.04}$ (Agrees: Non-diabetic).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth,keepaspectratio]{case_confident_healthy.png}
    \caption[Diagnostic Explanation for Confident Healthy Case]{Diagnostic explanation for Confident Healthy Patient \#29, governed by Rule 1.}
    \label{fig:case_confident_healthy}
\end{figure}

Figure~\ref{fig:case_confident_healthy} confirms that Patient \#29 belongs squarely to the metabolically healthy cohort. Rule~1 (Low Glucose $\times$ Low BMI $\times$ Low Age $\times$ Low Preg) activates with normalized weight $\bar{w}_1 = \mathbf{0.99}$ and match strength $\mathcal{S}_{\text{match}} = 1.00$, indicating that normal fasting glucose, young age, and lean physique guarantee negligible diabetes risk.

\section{Case Studies 3 and 4: Boundary Ambiguity and Model Disagreement}

Table~\ref{tab:case_ambiguity_summary} contrasts Patient \#75 (Borderline Ambiguity) and Patient \#89 (Surrogate Disagreement).

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.80}
\begin{tabularx}{\textwidth}{>{\bfseries\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X}
\toprule
\textbf{Clinical Metric} & \textbf{Patient \#75 (Borderline Case)} & \textbf{Patient \#89 (Surrogate Disagreement)} \\
\midrule
Clinical Profile & Glucose 119 mg/dL, BMI 27.2, Age 33, Preg 6 & Glucose 114 mg/dL, BMI 44.0, Age 27, Preg 0 \\
True Outcome & 1 (Diabetic) & 0 (Non-diabetic) \\
Black-Box MLP & $\hat{y}_{\text{NN}} = \mathbf{0.50}$ (Boundary uncertainty) & $\hat{y}_{\text{NN}} = \mathbf{0.26}$ (Classified Non-diabetic) \\
Neuro-Fuzzy Surrogate & $y = \mathbf{0.25}$ (Classified Non-diabetic) & $y = \mathbf{0.58}$ (\textbf{Classified Diabetic --- Disagrees}) \\
Match Strength ($\mathcal{S}_{\text{match}}$) & 0.62 (Moderate rule activation) & 0.38 (Sub-threshold activation) \\
Dominant Rule & \textbf{Rule 2} ($\bar{w}_2 = 0.61$, Med Glu $\times$ Med Preg) & \textbf{Rule 3} ($\bar{w}_3 = 0.52$, Med Glu $\times$ High BMI) \\
\midrule
Clinical Interpretation & Multiparity elevates Rule 2 firing, but lean BMI (27.2) keeps composite risk below 0.50. & Severe adiposity (BMI 44.0) triggers Rule 3 in the surrogate, while the MLP discounts adiposity in young nulliparous women. \\
\bottomrule
\end{tabularx}
\caption[Comparative Ambiguity and Disagreement Analysis]{Comparative case analysis of borderline ambiguity (Patient \#75) and model disagreement (Patient \#89).}
\label{tab:case_ambiguity_summary}
\end{table}

\section{Case Study 5: Atypical Patient and Weak-Match Alert (Patient \#108)}

Patient \#108 illustrates the safety protection provided by the \textbf{Weak-Match Diagnostic Flag}: Glucose = 194 mg/dL, BMI = 26.0 $\text{kg/m}^2$, Age = 67 yrs, Pregnancies = 8. True outcome: 0. Black box: $\hat{y}_{\text{NN}} = 0.73$; Neuro-fuzzy: $y = 0.80$.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth,keepaspectratio]{case_unusual_patient.png}
    \caption[Atypical Patient Triggering Weak-Match Alert]{Atypical Patient \#108 triggering the Weak-Match diagnostic warning alert.}
    \label{fig:case_unusual_patient}
\end{figure}

Figure~\ref{fig:case_unusual_patient} highlights the critical safety warning triggered for Patient \#108. The patient presents marked hyperglycemia (194 mg/dL) and advanced age (67 yrs), but lean body mass (BMI 26.0). Because every high-glucose rule (Rule 4) requires elevated BMI, raw rule firing collapses ($w_4 \approx 0.00$). While Layer~3 normalization artificially inflates relative weights, our safety diagnostic detects that $\mathcal{S}_{\text{match}} = \max w_r = \mathbf{0.00} < 0.20$, instantly warning clinicians: \textit{``WARNING: Atypical physiological outlier. Firing strength is 0.00 ($< 0.20$). Explanation classified as UNRELIABLE due to out-of-distribution input.''} This mechanism prevents silent automated diagnostic failure.
"""
