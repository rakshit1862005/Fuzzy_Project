# Professional Academic Front Matter (PEC Institutional Format - Electrical Engineering Dept)

FRONT_MATTER = r"""
\pagenumbering{roman}
\pagestyle{plain}

% ==============================================================================
% 1. COVER PAGE / TITLE PAGE (Centered institutional format with Large PEC Logo)
% ==============================================================================
\begin{titlepage}
    \begin{center}
    \vspace*{-0.5cm}
    {\fontsize{15}{18}\selectfont \bfseries PUNJAB ENGINEERING COLLEGE}\\[0.12cm]
    {\fontsize{13}{16}\selectfont \bfseries (DEEMED TO BE UNIVERSITY), CHANDIGARH}\\[0.12cm]
    {\fontsize{11}{14}\selectfont \bfseries DEPARTMENT OF ELECTRICAL ENGINEERING}\\[0.45cm]
    
    \includegraphics[width=3.6cm,keepaspectratio]{pec_logo.png}\\[0.45cm]
    
    {\fontsize{14}{17}\selectfont \bfseries PROJECT REPORT}\\[0.35cm]
    
    {\fontsize{15}{19}\selectfont \bfseries EXPLAINING NEURAL NETWORK DECISIONS IN\\ DIABETES DIAGNOSIS USING FUZZY LOGIC RULES}\\[0.45cm]
    
    {\fontsize{11}{14}\selectfont \textit{Course: Fuzzy Logic and Neural Networks (Course Code: EEO1003)}}\\[0.45cm]
    
    {\fontsize{11}{14}\selectfont \textbf{Under the Guidance of:}}\\[0.10cm]
    {\fontsize{11}{14}\selectfont \bfseries Dr.~Rintu Khanna \quad \& \quad Dr.~Hemant Kumar}\\[0.08cm]
    {\fontsize{10}{13}\selectfont Department of Electrical Engineering}\\[0.08cm]
    {\fontsize{10}{13}\selectfont Punjab Engineering College (Deemed to be University), Chandigarh}\\[1.20cm]
    
    {\fontsize{11}{14}\selectfont \textbf{Submitted by:}}\\[0.25cm]
    \noindent\begin{tabular*}{\linewidth}{@{}l @{\extracolsep{\fill}} r@{}}
        \textbf{Rakshit} (23106017) & \textbf{Aarav} (23106023) \\[0.15cm]
        \textbf{Aryan Garg} (23106024) & \textbf{Amish Bansal} (23103087) \\
    \end{tabular*}\\[0.60cm]
    
    \vfill
    {\fontsize{11}{14}\selectfont \bfseries Academic Year: 2026--2027}
    \end{center}
\end{titlepage}

% ==============================================================================
% 2. CANDIDATE'S DECLARATION (Page 2)
% ==============================================================================
\clearpage
\addcontentsline{toc}{chapter}{Candidate's Declaration}
\begin{center}
    \vspace*{0.2cm}
    {\fontsize{15}{18}\selectfont \bfseries CANDIDATE'S DECLARATION}\\[0.8cm]
\end{center}

\noindent
We hereby declare that the project work entitled \textbf{``Explaining Neural Network Decisions in Diabetes Diagnosis Using Fuzzy Logic Rules''} is an authentic record of our own engineering work carried out at \textbf{Punjab Engineering College (Deemed to be University), Chandigarh} as per the requirements of the course \textbf{Fuzzy Logic and Neural Networks (Course Code: EEO1003)} offered by the \textbf{Department of Electrical Engineering}, under the guidance and supervision of \textbf{Dr.~Rintu Khanna} and \textbf{Dr.~Hemant Kumar}.

\vspace{0.4cm}
\noindent
We further declare that the information, experimental pipelines, and results described in this report have been developed from authentic scientific sources, and we have not submitted this project report to this or any other institution for the award of any diploma or degree examination.

\vspace{1.2cm}
\noindent
\textbf{Date:} \today \hfill \textbf{Place:} Punjab Engineering College, Chandigarh

\vspace{2.2cm}
\noindent
\begin{tabular*}{\linewidth}{@{}l @{\extracolsep{\fill}} r@{}}
    \underline{\hspace{4.8cm}} & \underline{\hspace{4.8cm}} \\
    \textbf{Rakshit} (23106017) & \textbf{Aarav} (23106023) \\[1.8cm]
    \underline{\hspace{4.8cm}} & \underline{\hspace{4.8cm}} \\
    \textbf{Aryan Garg} (23106024) & \textbf{Amish Bansal} (23103087) \\
\end{tabular*}

% ==============================================================================
% 3. CERTIFICATE (Page 3 - Two Professors, No HOD)
% ==============================================================================
\clearpage
\addcontentsline{toc}{chapter}{Certificate}
\begin{center}
    \vspace*{-0.3cm}
    {\fontsize{14}{17}\selectfont \bfseries PUNJAB ENGINEERING COLLEGE}\\[0.08cm]
    {\fontsize{12}{15}\selectfont \bfseries (DEEMED TO BE UNIVERSITY), CHANDIGARH}\\[0.08cm]
    {\fontsize{11}{14}\selectfont \bfseries Department of Electrical Engineering}\\[0.30cm]
    \includegraphics[width=2.5cm,keepaspectratio]{pec_logo.png}\\[0.30cm]
    {\fontsize{15}{18}\selectfont \bfseries CERTIFICATE}\\[0.50cm]
\end{center}

\noindent
It is Certified that the Project work entitled \textbf{``Explaining Neural Network Decisions in Diabetes Diagnosis Using Fuzzy Logic Rules''} submitted by \textbf{Rakshit (23106017)}, \textbf{Aarav (23106023)}, \textbf{Aryan Garg (23106024)}, and \textbf{Amish Bansal (23103087)}, for the fulfilment of the course \textbf{Fuzzy Logic and Neural Networks (EEO1003)} offered by the \textbf{Department of Electrical Engineering, Punjab Engineering College (Deemed to be University)} during the academic year 2026--2027 is an original work carried out by the students under our supervision and guidance. This work has not framed the basis for the award of any other degree, diploma, or certificate. All the work related to the project has been carried out by these candidates themselves. The approach towards the subject has been sincere and scientific.

\vspace{1.2cm}
\noindent
\textbf{Date:} \today

\vspace{2.2cm}
\noindent
\begin{tabular*}{\linewidth}{@{}l @{\extracolsep{\fill}} r@{}}
    \begin{tabular}[t]{@{}l@{}}
        \underline{\hspace{5.5cm}} \\
        \textbf{Dr.~Rintu Khanna} \\
        (Professor) \\
        Department of Electrical Engineering \\
        Punjab Engineering College \\
        (Deemed to be University)
    \end{tabular} &
    \begin{tabular}[t]{@{}l@{}}
        \underline{\hspace{5.5cm}} \\
        \textbf{Dr.~Hemant Kumar} \\
        (Assistant Professor) \\
        Department of Electrical Engineering \\
        Punjab Engineering College \\
        (Deemed to be University)
    \end{tabular}
\end{tabular*}

% ==============================================================================
% 4. ACKNOWLEDGEMENT (Page 4)
% ==============================================================================
\clearpage
\addcontentsline{toc}{chapter}{Acknowledgement}
\begin{center}
    \vspace*{0.2cm}
    {\fontsize{15}{18}\selectfont \bfseries ACKNOWLEDGEMENT}\\[0.5cm]
\end{center}

\noindent
We would like to take this opportunity to express our sincere gratitude to our college \textbf{Punjab Engineering College (Deemed to be University), Chandigarh} and the \textbf{Department of Electrical Engineering} for offering the interdisciplinary course \textit{Fuzzy Logic and Neural Networks (EEO1003)} and providing us the opportunity to work on this project.

\vspace{0.25cm}
\noindent
We are immensely grateful to our project mentors \textbf{Dr.~Rintu Khanna} and \textbf{Dr.~Hemant Kumar} (Department of Electrical Engineering) whose continuous guidance, structured lectures, technical support, and valuable suggestions helped us develop and evaluate this neuro-fuzzy system. Their insights into fuzzy inference systems, membership function design, and neural networks were invaluable.

\vspace{0.25cm}
\noindent
We express our sincere thanks to the Project Evaluation Committee for their constructive feedback and guidelines for the betterment of this project report.

\vspace{0.25cm}
\noindent
We convey our deep sense of gratitude to all teaching and non-teaching staff of the Department of Electrical Engineering for their constant encouragement and assistance. We also extend appreciation to the curators of the benchmark Pima Indians Diabetes Database from the UCI Machine Learning Repository.

\vspace{0.25cm}
\noindent
Finally, we wish to express our heartfelt thanks to our families and friends for their constant moral support and encouragement throughout our engineering education.

\vspace{1.5cm}
\noindent
\begin{tabular*}{\linewidth}{@{}l @{\extracolsep{\fill}} r@{}}
    \textbf{Rakshit} (23106017) & \textbf{Aarav} (23106023) \\[0.6cm]
    \textbf{Aryan Garg} (23106024) & \textbf{Amish Bansal} (23103087) \\
\end{tabular*}

% ==============================================================================
% 5. ABSTRACT (Page 5)
% ==============================================================================
\clearpage
\addcontentsline{toc}{chapter}{Abstract}
\begin{center}
    \vspace*{0.1cm}
    {\fontsize{15}{18}\selectfont \bfseries ABSTRACT}\\[0.35cm]
\end{center}

\noindent
In healthcare applications, artificial neural networks achieve strong classification performance but operate as opaque black boxes. Because a neural network transforms input features through layers of non-linear weights and activations, doctors cannot see why a particular patient was classified as high risk. This opacity makes clinicians hesitant to trust automated predictions, especially when making critical medical decisions.

This project presents an interpretable neuro-fuzzy framework for explaining black-box neural network decisions in diabetes diagnosis. Developed for the course \textit{Fuzzy Logic and Neural Networks (EEO1003)}, the system bridges foundational neural networks and fuzzy inference systems: training an Artificial Neural Network (Multi-Layer Perceptron) using backpropagation, and extracting a Takagi-Sugeno-Kang (TSK) Fuzzy Inference System using shouldered Gaussian membership functions and Fuzzy C-Means (FCM) clustering.

Using the benchmark Pima Indians Diabetes Database ($N=768$, 8 clinical features), we first train a regularized MLP neural network ($8 \to 8 \to 1$) that achieves 70.1\% test accuracy and 0.797 ROC-AUC. Rather than predicting ground-truth labels directly, our neuro-fuzzy surrogate is trained to imitate the probability scores emitted by the neural network. To avoid rule explosion ($3^8 = 6{,}561$ rules), we apply Permutation Feature Importance to select the four primary clinical drivers: Glucose, BMI, Age, and Pregnancies. An 8-rule TSK fuzzy system is initialized from patient data via Fuzzy C-Means clustering.

The membership functions (Low, Medium, High) are tuned using hybrid learning: least squares solves the risk outputs in one step, while gradient descent fine-tunes the membership curves under semantic constraints (keeping Low $<$ Medium $<$ High). A backward pruning algorithm then eliminates redundant conditions and rules. The final model achieves \textbf{92.2\% test fidelity} to the neural network using only \textbf{4 readable IF-THEN rules and 12 conditions}. On confident predictions, fidelity reaches \textbf{98.3\%}. Any prediction can be verified by hand in 30 seconds using simple arithmetic, supported by an outlier detection metric ($\mathcal{S}_{\text{match}}$) and an interactive web laboratory with live feature sliders.

\vspace{0.35cm}
\noindent \textbf{Keywords:} Neural Networks, Fuzzy Logic, ANFIS, Rule Extraction, Explainable AI (XAI), Fuzzy C-Means, Model Interpretability, Clinical Decision Support.

% ==============================================================================
% 6. TABLE OF CONTENTS (Page 6 - Elegant Layout with Dotted Leaders)
% ==============================================================================
\clearpage
\begingroup
\linespread{0.95}\selectfont
\small
\setlength{\parskip}{0pt}
\setcounter{tocdepth}{0}
\makeatletter
\def\@makeschapterhead#1{%
  \vspace*{2\p@}%
  {\parindent \z@ \centering \normalfont
    {\fontsize{14}{17}\selectfont \bfseries TABLE OF CONTENTS}\par\nobreak
    \vskip 10\p@
  }}
\makeatother
\tableofcontents
\endgroup

% ==============================================================================
% 7. LIST OF FIGURES (Page 7 - Dedicated Page with Generous Spacing)
% ==============================================================================
\clearpage
\addcontentsline{toc}{chapter}{List of Figures}
\begingroup
\linespread{1.15}\selectfont
\setlength{\parskip}{0pt}
\makeatletter
\def\@makeschapterhead#1{%
  \vspace*{2\p@}%
  {\parindent \z@ \centering \normalfont
    {\fontsize{14}{17}\selectfont \bfseries \MakeUppercase{#1}}\par\nobreak
    \vskip 14\p@
  }}
\makeatother
\setcounter{tocdepth}{1}
\listoffigures
\endgroup

% ==============================================================================
% 8. LIST OF TABLES (Page 8 - Dedicated Page with Generous Spacing)
% ==============================================================================
\clearpage
\addcontentsline{toc}{chapter}{List of Tables}
\begingroup
\linespread{1.10}\selectfont
\setlength{\parskip}{0pt}
\makeatletter
\def\@makeschapterhead#1{%
  \vspace*{2\p@}%
  {\parindent \z@ \centering \normalfont
    {\fontsize{14}{17}\selectfont \bfseries \MakeUppercase{#1}}\par\nobreak
    \vskip 14\p@
  }}
\makeatother
\setcounter{tocdepth}{1}
\listoftables
\endgroup

% ==============================================================================
% 9. LIST OF ABBREVIATIONS & NOTATION (Page 9)
% ==============================================================================
\clearpage
\addcontentsline{toc}{chapter}{List of Abbreviations and Mathematical Notation}
\begin{center}
    {\fontsize{14}{17}\selectfont \bfseries LIST OF ABBREVIATIONS \& MATHEMATICAL NOTATION}\\[0.25cm]
\end{center}

{\linespread{0.95}\selectfont
\begin{center}
{\fontsize{11}{14}\selectfont \bfseries Acronyms}\\[0.15cm]
\footnotesize
\begin{tabularx}{\textwidth}{l X}
\toprule
\textbf{Acronym} & \textbf{Full Definition} \\
\midrule
\textbf{AI} & Artificial Intelligence \\
\textbf{ANFIS} & Adaptive Network-based (or Neuro-) Fuzzy Inference System \\
\textbf{ANN} & Artificial Neural Network \\
\textbf{AUC} & Area Under the Curve \\
\textbf{BCE} & Binary Cross-Entropy \\
\textbf{BMI} & Body Mass Index ($\text{kg/m}^2$) \\
\textbf{CDSS} & Clinical Decision Support System \\
\textbf{CV} & Cross-Validation \\
\textbf{DPF} & Diabetes Pedigree Function \\
\textbf{FCM} & Fuzzy C-Means \\
\textbf{FIS} & Fuzzy Inference System \\
\textbf{LIME} & Local Interpretable Model-agnostic Explanations \\
\textbf{MAE} & Mean Absolute Error \\
\textbf{MLP} & Multi-Layer Perceptron \\
\textbf{MSE} & Mean Squared Error \\
\textbf{PFI} & Permutation Feature Importance \\
\textbf{PIDD} & Pima Indians Diabetes Database \\
\textbf{ReLU} & Rectified Linear Unit ($\max(0, z)$) \\
\textbf{RLS} & Regularized Least Squares \\
\textbf{ROC} & Receiver Operating Characteristic \\
\textbf{SHAP} & SHapley Additive exPlanations \\
\textbf{SSE} & Server-Sent Events \\
\textbf{TSK} & Takagi-Sugeno-Kang (Fuzzy Model) \\
\textbf{XAI} & Explainable Artificial Intelligence \\
\bottomrule
\end{tabularx}
\end{center}

\vspace{0.15cm}
\begin{center}
{\fontsize{11}{14}\selectfont \bfseries Mathematical Symbols and Dimensions}\\[0.15cm]
\footnotesize
\begin{tabularx}{\textwidth}{l X l}
\toprule
\textbf{Symbol} & \textbf{Mathematical Meaning} & \textbf{Project Value} \\
\midrule
$N_{\text{train}}$ & Number of real training examples & 460 \\
$N_{\text{val}}, N_{\text{test}}$ & Real validation and holdout test examples & 154 each \\
$N_{\text{synth}}$ & Synthetic oracle neighborhood samples & 2,000 \\
$d, k$ & Raw feature count ($d=8$) and selected salient features ($k=4$) & 8, 4 \\
$m, R$ & Linguistic terms ($m=3$) and rule count (8 full $\to$ 4 pruned) & 3, 4 \\
$c_{jt}, \sigma_{jt}$ & Gaussian membership center and width parameters & Premise parameters \\
$\mu_{jt}(\tilde{x}_j)$ & Membership grade of input $j$ in linguistic term $t$ & $[0, 1]$ \\
$w_r, \bar{w}_r$ & Raw and normalized firing strength of fuzzy rule $r$ & $\sum \bar{w}_r = 1$ \\
$p_r, y(\tilde{x})$ & Consequent risk scalar and defuzzified surrogate prediction & $[0, 1]$ \\
$t_i, \mathcal{F}$ & Black-box continuous probability and empirical test fidelity & $f_{\text{NN}}(x_i), \ 92.2\%$ \\
\bottomrule
\end{tabularx}
\end{center}
}

% Switch to main body numbering and headers
\clearpage
\pagenumbering{arabic}
\pagestyle{fancy}
"""
