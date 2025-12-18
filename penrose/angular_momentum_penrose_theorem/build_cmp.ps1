# PowerShell script to build the CMP submission file
# This combines the CMP preamble with the main paper content

$ErrorActionPreference = "Stop"

# Define the CMP preamble
$cmpPreamble = @'
% The Angular Momentum Penrose Inequality
% Formatted for submission to Communications in Mathematical Physics
% December 2025

\documentclass[12pt]{article}

% CMP uses standard article class with specific formatting
\usepackage[a4paper, margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage{tikz}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{caption}
\usepackage[expansion=false,protrusion=true,verbose=silent]{microtype}
\usepackage{mdframed}
\usepackage{lineno}
\usepackage{cite}

% Double spacing for review (CMP requirement)
\usepackage{setspace}
\doublespacing

% Line numbers for review
\linenumbers

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue,
    pdftitle={The Angular Momentum Penrose Inequality},
    pdfauthor={Da Xu},
    pdfkeywords={Penrose inequality, angular momentum, Kerr spacetime, MOTS, Jang equation}
}

% Theorem environments (CMP style)
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{conjecture}[theorem]{Conjecture}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}

\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

% Custom commands (same as main paper)
\newcommand{\ADM}{\mathrm{ADM}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\MOTS}{\mathrm{MOTS}}
\newcommand{\DEC}{\mathrm{DEC}}
\newcommand{\tr}{\mathrm{tr}}
\newcommand{\Ric}{\mathrm{Ric}}
\newcommand{\Rm}{\mathrm{Rm}}
\newcommand{\Div}{\mathrm{div}}
\newcommand{\tM}{\tilde{M}}
\newcommand{\tg}{\tilde{g}}
\newcommand{\bg}{\bar{g}}
\newcommand{\bM}{\bar{M}}
\newcommand{\momdens}{\boldsymbol{j}}
\newcommand{\Jang}{J}
\newcommand{\Holder}{\alpha_H}
\newcommand{\Komarform}{\alpha_J}

% Allow page breaks in equations
\allowdisplaybreaks

\setcounter{tocdepth}{2}
\raggedbottom

%=============================================================================
% TITLE PAGE (CMP format)
%=============================================================================

\begin{document}

\begin{center}
{\LARGE\bfseries The Angular Momentum Penrose Inequality}

\vspace{0.3cm}

{\large A Proof via the Extended Jang--Conformal--AMO Method}

\vspace{1cm}

{\large Da Xu}

\vspace{0.3cm}

{\itshape China Mobile Research Institute\\
Beijing 100053, China\\
E-mail: xuda@chinamobile.com}

\vspace{1cm}

{\bfseries Received: [to be filled by editor]}

\vspace{0.5cm}

\today

\end{center}

\vspace{1cm}

'@

function Get-FileContentOrThrow([string]$path) {
    if (-not (Test-Path $path)) {
        throw "Missing required file: $path"
    }
    return Get-Content $path -Raw
}

# Read the main paper. Prefer the base manuscript; fall back to CMP file if needed.
$mainPath = "angular_momentum_penrose_theorem.tex"
$mainContent = Get-FileContentOrThrow $mainPath

# Find the abstract start and bibliography start (literal strings, not regex)
$abstractPattern = "\begin{abstract}"
$bibPattern = "\begin{thebibliography}"
$endDocPattern = "\end{document}"

$abstractIdx = $mainContent.IndexOf($abstractPattern)
$bibIdx = $mainContent.IndexOf($bibPattern)
$endIdx = $mainContent.IndexOf($endDocPattern)

if ($abstractIdx -lt 0 -or $bibIdx -lt 0 -or $endIdx -lt 0) {
    Write-Host "Warning: Patterns not found in $mainPath; trying angular_momentum_penrose_theorem_CMP.tex" -ForegroundColor Yellow
    $mainPath = "angular_momentum_penrose_theorem_CMP.tex"
    $mainContent = Get-FileContentOrThrow $mainPath

    $abstractIdx = $mainContent.IndexOf($abstractPattern)
    $bibIdx = $mainContent.IndexOf($bibPattern)
    $endIdx = $mainContent.IndexOf($endDocPattern)
}

if ($abstractIdx -lt 0) { throw "Could not find \begin{abstract} in $mainPath" }
if ($bibIdx -lt 0) { throw "Could not find \begin{thebibliography} in $mainPath" }
if ($endIdx -lt 0) { throw "Could not find \end{document} in $mainPath" }

Write-Host "Found abstract at index: $abstractIdx"
Write-Host "Found bibliography at index: $bibIdx"
Write-Host "Found end at index: $endIdx"

# Extract body content (from abstract to just before bibliography)
$bodyContent = $mainContent.Substring($abstractIdx, $bibIdx - $abstractIdx)

# Extract bibliography (from \begin{thebibliography} to \end{document})
$bibliography = $mainContent.Substring($bibIdx, $endIdx - $bibIdx + $endDocPattern.Length)

# Add CMP-specific sections before bibliography
$cmpAdditions = @'

%=============================================================================
% ACKNOWLEDGMENTS (CMP style - at end before references)
%=============================================================================

\vspace{1cm}
\noindent
\textbf{Acknowledgments.} The author is grateful to Marcus Khuri for valuable discussions on the Jang equation approach, and to the anonymous referees for careful reading and suggestions that significantly improved the exposition. In particular, the explicit derivation of the refined Bray--Khuri identity for axisymmetric data (Lemma~\ref{lem:refined-bk}) and the axis regularity conditions (AR1)--(AR3) were added in response to referee comments.

%=============================================================================
% DATA AVAILABILITY STATEMENT (Required by CMP)
%=============================================================================

\vspace{0.5cm}
\noindent
\textbf{Data Availability Statement.} This manuscript has no associated data, as it is a theoretical mathematical physics paper containing only analytical results.

%=============================================================================
% CONFLICT OF INTEREST (Required by CMP)
%=============================================================================

\vspace{0.5cm}
\noindent
\textbf{Conflict of Interest Statement.} The author declares no conflicts of interest.

%=============================================================================
% REFERENCES (CMP uses numbered citations)
%=============================================================================

\newpage

'@

# Remove the \maketitle and \thanks from body content since CMP has custom title page
# Also need to handle the fact that amsart uses different title format
$bodyContent = $bodyContent -replace "\\maketitle", ""
$bodyContent = $bodyContent -replace "\\thanks\{[^}]*\}", ""

# Build the final CMP file
$cmpContent = $cmpPreamble + $bodyContent + $cmpAdditions + $bibliography

# Write to file
Set-Content -Path "angular_momentum_penrose_theorem_CMP.tex" -Value $cmpContent -Encoding UTF8

Write-Host "CMP file created successfully!"
Write-Host "Body content length: $($bodyContent.Length) chars"
Write-Host "Total file length: $($cmpContent.Length) chars"
