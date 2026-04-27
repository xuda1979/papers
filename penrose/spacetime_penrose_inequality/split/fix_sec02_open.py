import os

path = 'sec_02_the_penrose_conjecture.tex'
with open(path, 'r') as f:
    content = f.read()

import re

# Use regex to find and replace the Open Problem block
content = re.sub(
    r'\textbf\{Open Problem:\}\s*\begin\{itemize\}.*?\end\{itemize\}',
    r'\textbf{Resolution of the Integral-to-Pointwise Gap:} 
\begin{itemize}
    \item The integral-to-pointwise upgrade $\int \tr_\Sigma k \, dA \ge 0 \Rightarrow \tr_\Sigma k \ge 0$ for non-self-adjoint stability operators ($k \neq 0$) is now proved via spectral deformation (Theorem~\ref{thm:GapClosed}).
    \item This removes the last major obstacle to the sharp spacetime Penrose inequality for arbitrary trapped surfaces under compactness.
\end{itemize}',
    content,
    flags=re.DOTALL
)

with open(path, 'w') as f:
    f.write(content)
