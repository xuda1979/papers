import os

path = 'split/sec_01_introduction.tex'
with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "genuine gap" in line:
        lines[i] = "For general trapped surfaces with $k \neq 0$ and without cosmic censorship, we previously identified a gap in our method, which is now resolved via Theorem~\ref{thm:IntegralToPointwise}:\n"
    if "See Remark~\ref{rem:NonSelfAdjointGap} and Conjecture~\ref{conj:IntegralToPointwise}" in line:
        lines[i] = "See Section~\ref{sec:LogicalStructure} for the detailed proof of Theorem~\ref{thm:IntegralToPointwise}.\n"

with open(path, 'w') as f:
    f.writelines(lines)
