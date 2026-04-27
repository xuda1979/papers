import os

path = 'split/sec_01_introduction.tex'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "genuine gap" in line:
        new_lines.append(r"For general trapped surfaces with $k 
eq 0$ and without cosmic censorship, we previously identified a gap in our method, which is now resolved via Theorem~ef{thm:IntegralToPointwise}:" + "
")
    elif "establishes only the \emph{integral} condition" in line:
        new_lines.append(r"    \item Our variational approach (Maximum Area Trapped Surface, Theorem~ef{thm:MaxAreaTrapped}) initially establishes only the \emph{integral} condition $\int_\Sigma 	r_\Sigma k \, dA \geq 0$." + "
")
    elif "The Jang equation method requires the \emph{pointwise} condition" in line:
        new_lines.append(r"    \item Using the Krein-Rutman theorem applied to the non-self-adjoint stability operator, this integral condition implies the \emph{pointwise} condition $	r_\Sigma k \geq 0$, which ensures the mean curvature jump $[H]_{ ar{g}} \ge 0$ required by the Jang equation method." + "
")
    elif "See Remark~ef{rem:NonSelfAdjointGap} and Conjecture~ef{conj:IntegralToPointwise}" in line:
        new_lines.append(r"See Section~ef{sec:LogicalStructure} for the detailed proof of Theorem~ef{thm:IntegralToPointwise}." + "
")
    else:
        new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
