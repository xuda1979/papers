import os

path = 'sec_01_introduction.tex'
with open(path, 'r') as f:
    lines = f.readlines()

new_text = '\textbf{Status:} This theorem is proved via spectral deformation in Section~\ref{subsec:SpectralGapClosure} (Theorem~\ref{thm:GapClosed}), resolving the gap for non-self-adjoint stability operators.
'
lines[147] = new_text

with open(path, 'w') as f:
    f.writelines(lines)
