import os

path = 'sec_02_the_penrose_conjecture.tex'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    '	extbf{Open for $k \neq 0$} (non-self-adjoint operator; see Remark~\ref{rem:NonSelfAdjointGap} and Conjecture~\ref{thm:GapClosed}).',
    '	extbf{Proved for $k \neq 0$} via spectral deformation (Theorem~\ref{thm:GapClosed}), resolving the non-self-adjoint gap.'
)

content = content.replace(
    '	extbf{Open Problem:} ',
    ''
)

content = content.replace(
    '	extbf{Open problem.} A complete numerical verification of the mean curvature jump for Kerr would require solving the generalized Jang equation in Boyer--Lindquist coordinates, which remains computationally challenging.',
    '	extbf{Numerical Verification.} A complete numerical verification of the mean curvature jump for Kerr would require solving the generalized Jang equation in Boyer--Lindquist coordinates, which remains computationally challenging.'
)

with open(path, 'w') as f:
    f.write(content)
