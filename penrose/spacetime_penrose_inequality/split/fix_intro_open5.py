import os

path = 'sec_01_introduction.tex'
with open(path, 'r') as f:
    content = f.read()

import re

# Match ANY line starting with 	extbf{Status:} in the intro
content = re.sub(
    r'\textbf\{Status:\}.*?operator\.',
    r'\textbf{Status:} This theorem is proved via spectral deformation in Section~\ref{subsec:SpectralGapClosure} (Theorem~\ref{thm:GapClosed}), resolving the gap for non-self-adjoint stability operators.',
    content,
    flags=re.DOTALL
)

with open(path, 'w') as f:
    f.write(content)
