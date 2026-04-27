import os

path = '/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_05_ricci_flow_inspired_monotonicity_formulas.tex'
with open(path, 'r') as f:
    content = f.read()

# Just rewrite the block directly
content = content.replace('	extbf{Current status:} The spectral deformation method (Theorem~', '\textbf{Current status:} The spectral deformation method (Theorem~\ref{thm:GapClosed}) completely resolves this gap by providing an integral-to-pointwise upgrade for the non-self-adjoint stability operator.
\end{remark}')
content = content.replace('ef{thm:GapClosed}) completely resolves this gap by providing an integral-to-pointwise upgrade for the non-self-adjoint stability operator.', '')
content = content.replace('d{remark}', '')

with open(path, 'w') as f:
    f.write(content)
