# Chapter 4: The Black-Box Neural Network: Design, Optimization and Feature Importance

CHAPTER_4 = r"""
\chapter{The Black-Box Neural Network: Design, Optimization and Feature Importance}

\section{Neural Network Architecture and Forward Pass Formulation}

To serve as the clinical black box, we construct a fully connected feedforward Multi-Layer Perceptron (MLP) mapping standardized 8-dimensional measurement vectors $x^{\text{std}} \in \mathbb{R}^8$ to continuous posterior probabilities $P(\text{Outcome} = 1 \mid x) \in [0, 1]$.

\subsection{Architectural Specification and Forward Propagation}
The architecture possesses an $8 \to 8 \to 1$ topology: an input layer of 8 standardized clinical variables, a hidden layer of 8 ReLU units ($L_1$), and an output layer of 1 logistic sigmoid unit ($L_2$).

Let $W^{[1]} \in \mathbb{R}^{8 \times 8}$ and $b^{[1]} \in \mathbb{R}^8$ denote hidden layer parameters. The hidden pre-activation $z^{[1]}$ and post-activation $h \in \mathbb{R}^8$ are:
\begin{equation}
    z^{[1]} = W^{[1]} x^{\text{std}} + b^{[1]}, \qquad h = \text{ReLU}\left(z^{[1]}\right) = \max\left(0, z^{[1]}\right)
\end{equation}
With output layer parameters $W^{[2]} \in \mathbb{R}^{1 \times 8}$ and $b^{[2]} \in \mathbb{R}^1$, scalar logit $z^{[2]}$ and posterior risk $\hat{y}_{\text{NN}}$ are:
\begin{equation}
    z^{[2]} = W^{[2]} h + b^{[2]}, \qquad \hat{y}_{\text{NN}} = \sigma\left(z^{[2]}\right) = \frac{1}{1 + e^{-z^{[2]}}} = P(\text{Outcome} = 1 \mid x)
\end{equation}
For discrete diagnosis, threshold $\tau = 0.5$ is applied: $\hat{Y}_{\text{NN}} = \mathbb{I}[\hat{y}_{\text{NN}} \ge 0.5]$.

\subsection{Parameter Complexity}
The total trainable continuous parameter count is:
\begin{equation}
    |\theta_{\text{NN}}| = (8 \times 8 + 8) + (8 \times 1 + 1) = 72 + 9 = \mathbf{81\text{ parameters}}
\end{equation}
Although 81 weights is compact compared to computer vision models, in tabular clinical diagnostics, 81 interconnected floating-point weights with non-linear ReLU gating already constitute an impenetrable black box. No clinician can inspect these 81 parameters and verify why a specific patient was classified as high risk.

\section{Optimization Objective and Training Dynamics}

The network is trained by minimizing $L_2$-regularized Binary Cross-Entropy (BCE) loss over $\mathcal{D}_{\text{train}}$:
\begin{equation}
    \mathcal{L}(\theta) = -\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\alpha}{2} \left( \|W^{[1]}\|_F^2 + \|W^{[2]}\|_2^2 \right)
\end{equation}
where $\alpha > 0$ denotes weight decay. We utilize Glorot uniform initialization and the Adam optimizer ($\eta = 0.001$, $\beta_1 = 0.9, \beta_2 = 0.999$, batch size $B = 200$).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth,keepaspectratio]{training_curve.png}
    \caption[Neural Network Training Loss Convergence]{Training loss convergence curve of the black-box neural network over 321 epochs under Adam optimization.}
    \label{fig:nn_training_curve}
\end{figure}

Figure~\ref{fig:nn_training_curve} illustrates the empirical convergence of training loss across epochs. As shown in the plot, cross-entropy drops sharply from 0.693 to approximately 0.480 within the first 40 epochs as the hidden representations adapt to primary glycemic and adiposity gradients. Thereafter, Adam optimization guides smooth, monotonic descent toward 0.441. Under early stopping ($\Delta \mathcal{L} < 10^{-4}$ over 10 consecutive epochs), training terminates stably at epoch 321 without oscillatory divergence or gradient saturation.

\section{Cross-Validation and Diagnostic Performance}

Topology and regularization were selected via \textbf{5-fold stratified cross-validation} across the 460 training patients, evaluating 16 combinations of hidden layer sizes and $\alpha \in \{10^{-3}, 10^{-2}, 10^{-1}, 1.0\}$. The single hidden layer of 8 neurons with $\alpha = 1.0$ achieved the highest mean ROC-AUC ($0.826 \pm 0.017$). Strong regularization ($\alpha = 1.0$) suppresses spurious weight fluctuations, yielding smooth posterior risk surfaces.

Following cross-validation, the selected MLP was evaluated on the held-out test partition ($N_{\text{test}} = 154$) alongside a reference $L_2$-regularized Logistic Regression baseline (Table~\ref{tab:diagnostic_performance}).

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.80}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X c c c c}
\toprule
\textbf{Model Evaluated} & \textbf{Test Accuracy} & \textbf{Test F1-Score} & \textbf{Test ROC-AUC} & \textbf{Parameters} \\
\midrule
\textbf{Black-Box MLP (8 $\to$ 8 $\to$ 1)} & \textbf{0.701} & \textbf{0.558} & \textbf{0.797} & 81 \\
Logistic Regression (Reference) & 0.708 & 0.545 & 0.817 & 9 \\
\bottomrule
\end{tabularx}
\caption[Diagnostic Test-Set Performance Comparison]{Diagnostic test-set performance comparison between the black-box neural network and logistic regression.}
\label{tab:diagnostic_performance}
\end{table}

The MLP achieves 70.1\% accuracy and 0.797 ROC-AUC, demonstrating performance parity with the linear baseline (70.8\%, 0.817). In small tabular biomedical datasets, linear models frequently match shallow networks. Crucially, the objective of this project is not to prove that neural networks beat linear models on PIDD, but to \textbf{explain deployed opaque neural networks} when institutional or engineering requirements dictate their deployment.

\section{Permutation Feature Importance and Saliency Selection}

To construct an interpretable neuro-fuzzy surrogate without suffering rule explosion ($3^8 = 6{,}561$ rules), we determine which inputs drive black-box inferences using \textbf{Permutation Feature Importance} (Breiman, 2001; Fisher et al., 2019). For each feature $j \in \{1, \dots, d\}$, column values are shuffled across validation samples, breaking target association. The resulting loss increase ($I(j) = L^{\text{perm}(j)} - L_{\text{baseline}}$) is averaged across $M=10$ random shuffles to compute normalized importance shares $S(j)$.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.75\textwidth,keepaspectratio]{feature_importance.png}
    \caption[Permutation Feature Importance Rankings]{Permutation feature importance rankings of the black-box neural network, identifying dominant clinical drivers.}
    \label{fig:feature_importance}
\end{figure}

\begin{table}[htbp]
\centering
\footnotesize
\renewcommand{\arraystretch}{0.75}
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{l r r X}
\toprule
\textbf{Clinical Feature} & \textbf{Loss Increase} & \textbf{Importance Share} & \textbf{Surrogate Selection Status} \\
\midrule
\texttt{Glucose} & 0.231 & 43.8\% & \textbf{Selected (Rank 1)} \\
\texttt{BMI} & 0.110 & 20.8\% & \textbf{Selected (Rank 2)} \\
\texttt{Age} & 0.051 & 9.7\% & \textbf{Selected (Rank 3)} \\
\texttt{Pregnancies} & 0.045 & 8.6\% & \textbf{Selected (Rank 4)} \\
\texttt{Pedigree} & 0.043 & 8.2\% & Excluded \\
\texttt{BloodPressure} & 0.024 & 4.5\% & Excluded \\
\texttt{Insulin} & 0.016 & 3.1\% & Excluded (Severe missingness) \\
\texttt{SkinThickness} & 0.007 & 1.3\% & Excluded (Severe missingness) \\
\bottomrule
\end{tabularx}
\caption[Permutation Feature Importance Attribution]{Permutation feature importance analysis of the trained black-box neural network.}
\label{tab:pfi_results}
\end{table}

Figure~\ref{fig:feature_importance} and Table~\ref{tab:pfi_results} display the resulting attribution hierarchy. As visually evidenced, plasma glucose dominates network behavior (43.8\% share), followed by BMI (20.8\%). Together with Age (9.7\%) and Pregnancies (8.6\%), the top four features account for \textbf{82.9\%} of total model attribution. The bottom features contribute minimally; notably, Insulin (3.1\%) and SkinThickness (1.3\%) are largely discounted by the network due to the high missingness documented in Chapter 3.

Accordingly, we isolate $k = 4$ salient features: \textbf{Glucose, BMI, Age, and Pregnancies}. Restricting antecedents to these 4 primary drivers reduces the dimension space by 50\%, aligns the surrogate with established medical pathology, and makes compact fuzzy rule extraction computationally tractable.
"""
