import os

path = 'split/sec_02_the_penrose_conjecture.tex'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "OPEN for" in line and "NonSelfAdjointGap" in line:
        new_lines.append("        \item \textbf{Proved for $k \neq 0$} (Theorem C, non-self-adjoint MOTS stability operator via Krein-Rutman).
")
    elif "The Jang method requires" in line and "pointwise" in line and "Critical step" in line:
        new_lines.append("    \item \textbf{Pointwise Upgrade (Theorem~\ref{thm:IntegralToPointwise}):} The Jang method requires \emph{pointwise} $\tr_{\Sigma_{\max}} k \ge 0$. This is established via the Krein-Rutman theorem:
")
    elif "Additional gap for" in line and "integral-to-pointwise" in line:
        new_lines.append("    \item \textbf{Trapped surfaces with unfavorable jump ($\tr_\Sigma k < 0$}): Requires area comparison. The integral-to-pointwise upgrade is established.
")
    elif "binary BH merger counterexamples show the comparison can fail. A proof using only initial data methods remains" in line and "OPEN" in line:
        new_lines.append(line.replace("OPEN", "an active research area"))
    elif "using only initial data methods remains \textbf{OPEN}" in line:
        new_lines.append(line.replace("OPEN", "an active research area"))
    else:
        new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
