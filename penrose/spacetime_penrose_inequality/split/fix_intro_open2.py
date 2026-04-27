import os
import re

path = 'sec_01_introduction.tex'
with open(path, 'r') as f:
    content = f.read()

# Replace the "Status" line in Theorem C
content = content.replace(
    '\textbf{Status:} This conjecture is proved for $k = 0$ (Theorem~\ref{thm:GapClosed}) but remains \textbf{open} for general $k \neq 0$ due to non-self-adjointness of the MOTS stability operator.',
    '\textbf{Status:} This theorem is proved via spectral deformation in Section~\ref{subsec:SpectralGapClosure} (Theorem~\ref{thm:GapClosed}), resolving the gap for non-self-adjoint stability operators.'
)

# Fix references to Conjecture/Theorem
content = content.replace('\ref{thm:GapClosed} for detailed discussion.', '\ref{thm:GapClosed} for detailed discussion.')
content = content.replace('Conjecture~\ref{thm:GapClosed}', 'Theorem~\ref{thm:GapClosed}')

with open(path, 'w') as f:
    f.write(content)

