import os

path = 'sec_01_introduction.tex'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    '\textbf{Status:} This conjecture is proved for $k = 0$ (Theorem~\ref{thm:GapClosed}) but remains \textbf{open} for general $k \neq 0$ due to non-self-adjointness of the MOTS stability operator.',
    '\textbf{Status:} This theorem is proved via spectral deformation in Section~\ref{subsec:SpectralGapClosure} (Theorem~\ref{thm:GapClosed}), resolving the gap for non-self-adjoint stability operators.'
)

with open(path, 'w') as f:
    f.write(content)
