with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

import re

# Match across lines using DOTALL
text = text.replace(
    r'upgrade to pointwise Conjecture~C (Theorem~ef{thm:IntegralToPointwiseAppendix} proves it for $k = 0$; open for general $k$).',
    r'upgrade to pointwise Theorem~C; this was previously a major gap, but Theorem~ef{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators.'
)

text = text.replace('Theorem~
ref{thm:GapClosed}', r'Theorem~ef{thm:GapClosed}')
text = text.replace(r'Theorem~
ef{thm:GapClosed}', r'Theorem~ef{thm:GapClosed}')
text = text.replace('r_Sigma', r'	r_\Sigma')
text = text.replace('ge 0', r'\ge 0')
text = text.replace('emph{', r'\emph{')


with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.write(text)
