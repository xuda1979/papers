import os

path = 'sec_01_introduction.tex'
with open(path, 'r') as f:
    content = f.read()

import re
content = re.sub(
    r'\textbf\{Status:\} This conjecture is proved for \$k = 0\$ \(Theorem~\ref\{thm:GapClosed\}\) but remains \textbf\{open\} for general \$k \neq 0\$ due to non-self-adjointness of the MOTS stability operator\.',
    r'\textbf{Status:} This theorem is proved via spectral deformation in Section~\ref{subsec:SpectralGapClosure} (Theorem~\ref{thm:GapClosed}), resolving the gap for non-self-adjoint stability operators.',
    content
)

with open(path, 'w') as f:
    f.write(content)
