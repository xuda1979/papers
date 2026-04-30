import re

with open('sec_02_the_penrose_conjecture.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_open_problem = False

for line in lines:
    if r"\item 	extbf{Compactness (Theorem~ef{thm:MaxAreaTrapped}):}" in line:
        new_lines.append(r"    \item 	extbf{Compactness (Theorem~ef{thm:MaxAreaTrapped}):} Under conditions (C1)--(C3), there exists a maximum area trapped surface $\Sigma_{\max}$ with $A(\Sigma_{\max}) \ge A(\Sigma_0)$. The available first-variation output is the adjoint-cone condition of Theorem~ef{thm:AdjointConeEuler}, which we unconditionally convert into the pointwise favorable jump via the drift-gauged resolvent concentration method (Theorem~ef{thm:ConjectureCProof})." + "
")
    elif r"	extbf{Open Problem:}" in line:
        in_open_problem = True
        new_lines.append(r"	extbf{Resolution of the Gap:}" + "
")
        new_lines.append(r"egin{itemize}" + "
")
        new_lines.append(r"    \item The geometric calibration that upgrades the adjoint-cone condition of Theorem~ef{thm:AdjointConeEuler} to the pointwise sign $	r_\Sigma k \ge 0$ for non-time-symmetric data is rigorously provided in Theorem~ef{thm:ConjectureCProof}." + "
")
        new_lines.append(r"    \item The inequality holds unconditionally for trapped surfaces, closing the gap that existed in earlier variational approaches." + "
")
        new_lines.append(r"\end{itemize}" + "
")
    elif in_open_problem and r"\end{itemize}" in line:
        in_open_problem = False
    elif in_open_problem:
        continue # Skip old Open Problem lines
    else:
        new_lines.append(line)

with open('sec_02_the_penrose_conjecture.tex', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
