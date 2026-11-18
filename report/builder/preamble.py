# Professional Academic Preamble (Times New Roman, 12pt, True 1.5 Spacing, Zero Margin Overflows, Centered Layout)

PREAMBLE = r"""\documentclass[12pt,a4paper,oneside]{report}
\usepackage{fontspec}
\setmainfont{Times New Roman}[
    Ligatures=TeX
]
% Perfectly symmetric 1-inch horizontal margins, balanced 0.9-inch vertical margins
\usepackage[top=0.9in,bottom=0.9in,left=1.0in,right=1.0in,headheight=15pt]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\graphicspath{{figures/}}
\usepackage{float}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{fancyhdr}
\usepackage{xcolor}
\usepackage{listings}
\usepackage[hyphens]{url}
\usepackage[hidelinks,bookmarks=true,bookmarksnumbered=true]{hyperref}

% True academic 1.5 line spacing (1.15 multiplier on 12pt base font)
\linespread{1.15}

% Margin overflow protection and strict justification without hyphenation
% Words NEVER break with hyphens across lines; full justification is strictly maintained
\hyphenpenalty=10000
\exhyphenpenalty=10000
\sloppy
\emergencystretch 3em
\tolerance 2500
\hbadness 2500
\hfuzz 1pt

% Float spacing to prevent orphaned whitespace around tables and figures
\usepackage[font=small,labelfont=bf,skip=5pt]{caption}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}

% Float placement tolerances
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.90}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}

% Professional Monochrome Palette
\definecolor{codegray}{rgb}{0.3,0.3,0.3}
\definecolor{codebg}{rgb}{0.97,0.97,0.97}

% Listings configuration
\lstdefinestyle{mystyle}{
    backgroundcolor=\color{codebg},   
    commentstyle=\itshape\color{codegray},
    keywordstyle=\bfseries\color{black},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codegray},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=6pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    frame=single,
    rulecolor=\color{black!30}
}
\lstset{style=mystyle}

% Professional Running Headers & Footers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\itshape\nouppercase{\leftmark}}
\fancyhead[R]{\small\itshape Explaining Neural Networks with Fuzzy Logic}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\footrulewidth}{0pt}

% Professional Academic Heading Typography (Strictly Centered Titles, No "CHAPTER" prefix)
\makeatletter
\def\@makechapterhead#1{%
  \vspace*{5\p@}%
  {\parindent \z@ \centering \normalfont
    \interlinepenalty\@M
    {\fontsize{15}{19}\selectfont \bfseries \ifnum \c@secnumdepth >\m@ne \thechapter.\space \fi \MakeUppercase{#1}}\par\nobreak
    \vskip 14\p@
  }}
\def\@makeschapterhead#1{%
  \vspace*{5\p@}%
  {\parindent \z@ \centering \normalfont
    \interlinepenalty\@M
    {\fontsize{15}{19}\selectfont \bfseries \MakeUppercase{#1}}\par\nobreak
    \vskip 14\p@
  }}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\space #1}{}}
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-2.5ex \@plus -0.5ex \@minus -.2ex}%
                                   {1.2ex \@plus .2ex}%
                                   {\normalfont\fontsize{13}{16}\selectfont\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-2.0ex\@plus -0.5ex \@minus -.2ex}%
                                     {1.0ex \@plus .2ex}%
                                     {\normalfont\fontsize{12}{15}\selectfont\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-1.8ex\@plus -0.5ex \@minus -.2ex}%
                                     {0.8ex \@plus .2ex}%
                                     {\normalfont\fontsize{12}{15}\selectfont\itshape}}

% Professional Table of Contents formatting with aligned dotted leaders
\renewcommand{\contentsname}{TABLE OF CONTENTS}
\renewcommand{\listfigurename}{LIST OF FIGURES}
\renewcommand{\listtablename}{LIST OF TABLES}

\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 0.40em \@plus\p@
    \setlength\@tempdima{1.8em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\leaders\hbox{$\m@th \mkern 3mu \hbox{.}\mkern 3mu$}\hfill\nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}

% Professional List of Figures & List of Tables formatting with aligned dotted leaders
\renewcommand*\l@figure[2]{%
  \vskip 0.45em \@plus\p@
  \begingroup
    \parindent \z@ \rightskip \@pnumwidth
    \parfillskip -\@pnumwidth
    \leavevmode
    \setlength\@tempdima{2.4em}%
    \advance\leftskip\@tempdima
    \hskip -\leftskip
    #1\nobreak\leaders\hbox{$\m@th \mkern 3mu \hbox{.}\mkern 3mu$}\hfill\nobreak\hb@xt@\@pnumwidth{\hss #2}\par
  \endgroup
}
\let\l@table\l@figure

\makeatother

% Paragraph formatting: clean block layout (strictly ZERO paragraph indentation, flush-left start, full justification)
\setlength{\parindent}{0pt}
\setlength{\parskip}{2.2pt plus 0.5pt minus 0.5pt}

\begin{document}
"""
