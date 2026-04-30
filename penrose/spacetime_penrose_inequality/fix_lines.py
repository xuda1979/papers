with open("/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_02_the_penrose_conjecture.tex", "r") as f:
    lines = f.readlines()

lines[114] = "    \item \textbf{Compactness (Theorem~\ref{thm:MaxAreaTrapped}):} Under conditions (C1)--(C3), there exists a maximum area trapped surface $\Sigma_{\max}$ with $A(\Sigma_{\max}) \ge A(\Sigma_0)$. The available first-variation output is the adjoint-cone condition of Theorem~\ref{thm:AdjointConeEuler}, which we unconditionally convert into the pointwise favorable jump via the drift-gauged resolvent concentration method (Theorem~\ref{thm:ConjectureCProof}).
"

lines[117] = "\textbf{Resolution of the Gap:}
"

lines[119] = "    \item The geometric calibration that upgrades the adjoint-cone condition of Theorem~\ref{thm:AdjointConeEuler} to the pointwise sign $\tr_\Sigma k \ge 0$ for non-time-symmetric data is rigorously provided in Theorem~\ref{thm:ConjectureCProof}.
"

lines[120] = "    \item The inequality holds unconditionally for trapped surfaces, closing the gap that existed in earlier variational approaches.
"

with open("/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_02_the_penrose_conjecture.tex", "w") as f:
    f.writelines(lines)
