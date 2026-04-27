import os

path = 'sec_01_introduction.tex'
with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if '\textbf{Status:} This conjecture is proved for $k = 0$' in line:
        lines[i] = '\textbf{Status:} This theorem is proved via spectral deformation in Section~\ref{subsec:SpectralGapClosure} (Theorem~\ref{thm:GapClosed}), resolving the gap for non-self-adjoint stability operators.
'

with open(path, 'w') as f:
    f.writelines(lines)
