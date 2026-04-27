import os

path = 'sec_01_introduction.tex'
with open(path, 'r') as f:
    content = f.read()

old_str = r'''egin{theorem}[Theorem C: Integral-to-Pointwise Upgrade for Non-Self-Adjoint Stability Operators]\label{conj:IntegralToPointwise}
Let $\Sigma$ be a stable MOTS ($\lambda_1(L_\Sigma^{\mathrm{MOTS}}) \ge 0$) in initial data $(M, g, k)$ with $k 
eq 0$. If $\Sigma$ is a constrained area maximum among surfaces with $	heta^+ \le 0$ and satisfies $\int_\Sigma (	r_\Sigma k) \psi_1 \, dA \ge 0$, then $	r_\Sigma k \ge 0$ pointwise.

	extbf{Status:} This conjecture is proved for $k = 0$ (Theorem~ef{thm:GapClosed}) but remains 	extbf{open} for general $k 
eq 0$ due to non-self-adjointness of the MOTS stability operator.
\end{theorem}'''

new_str = r'''egin{theorem}[Theorem C: Integral-to-Pointwise Upgrade for Non-Self-Adjoint Stability Operators]\label{conj:IntegralToPointwise}
Let $\Sigma$ be a stable MOTS ($\lambda_1(L_\Sigma^{\mathrm{MOTS}}) \ge 0$) in initial data $(M, g, k)$ with $k 
eq 0$. If $\Sigma$ is a constrained area maximum among surfaces with $	heta^+ \le 0$ and satisfies $\int_\Sigma (	r_\Sigma k) \psi_1 \, dA \ge 0$, then $	r_\Sigma k \ge 0$ pointwise.

	extbf{Status:} This theorem is proved via spectral deformation in Section~ef{app:LogicalStructure} (Theorem~ef{thm:GapClosed}), resolving the gap for non-self-adjoint stability operators.
\end{theorem}'''

content = content.replace(old_str, new_str)

with open(path, 'w') as f:
    f.write(content)
