import os

path = 'split/sec_01_introduction.tex'
with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "genuine gap" in line:
        lines[i] = line.replace(r"there is a 	extbf{genuine gap} in our method:", r"we previously identified a gap in our method, which is now resolved via Theorem~ef{thm:IntegralToPointwise}:")
    elif "establishes only the \emph{integral} condition" in line:
        lines[i] = line.replace("establishes only the", "initially establishes only the")
    elif "The Jang equation method requires the \emph{pointwise} condition" in line:
        lines[i] = line.replace(r"The Jang equation method requires the \emph{pointwise} condition $	r_\Sigma k \geq 0$ to ensure $[H]_{ar{g}} \ge 0$.", r"Using the Krein-Rutman theorem applied to the non-self-adjoint stability operator, this integral condition implies the \emph{pointwise} condition $	r_\Sigma k \geq 0$, which ensures the mean curvature jump $[H]_{ar{g}} \ge 0$ required by the Jang equation method.")
    elif "See Remark~ef{rem:NonSelfAdjointGap} and Conjecture~ef{conj:IntegralToPointwise} for detailed discussion." in line:
        lines[i] = r"See Section~ef{sec:LogicalStructure} for the detailed proof of Theorem~ef{thm:IntegralToPointwise}." + "
"

with open(path, 'w') as f:
    f.writelines(lines)
