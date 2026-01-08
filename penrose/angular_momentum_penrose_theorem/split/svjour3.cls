% Springer Nature svjour3 document class
% Simplified version for Peking Mathematical Journal submission
% For full version, download from: https://www.springer.com/gp/livingreviews/latex-templates

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{svjour3}[2023/01/01 v3.2 Springer journal article class]

% Base class
\DeclareOption{smallextended}{\PassOptionsToClass{11pt}{article}}
\DeclareOption{twocolumn}{\PassOptionsToClass{twocolumn}{article}}
\DeclareOption{referee}{\PassOptionsToClass{12pt}{article}\AtBeginDocument{\doublespacing\linenumbers}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass{article}

% Required packages
\RequirePackage{geometry}
\geometry{a4paper,top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm}

% Journal name command
\newcommand{\journalname}[1]{\gdef\@journalname{#1}}
\gdef\@journalname{Peking Mathematical Journal}

% Author institute
\newcommand{\institute}[1]{\gdef\@institute{#1}}
\gdef\@institute{}
\newcommand{\email}[1]{\texttt{#1}}
\newcommand{\at}{\\}

% Redefine maketitle
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE\bfseries\@title\par}
    \vspace{1em}
    {\large\@author\par}
    \vspace{0.5em}
    {\small\itshape\@institute\par}
    \vspace{0.5em}
    {\small\@date\par}
  \end{center}
  \vspace{1em}
}

% Keywords and subject classification
\newcommand{\keywords}[1]{\par\medskip\noindent\textbf{Keywords:} #1}
\newcommand{\subclass}[1]{\par\smallskip\noindent\textbf{Mathematics Subject Classification (2020):} #1}

% Acknowledgements environment
\newenvironment{acknowledgements}{%
  \section*{Acknowledgements}
}{}

% Springer theorem styles
\newcommand{\spnewtheorem}[4]{%
  \theoremstyle{plain}%
  \newtheorem{#1}{#2}[section]%
}

\endinput
