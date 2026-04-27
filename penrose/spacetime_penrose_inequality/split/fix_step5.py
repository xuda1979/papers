with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    lines = f.readlines()

new_text = "\textbf{Step 5 (Non-Self-Adjoint Case, $k \neq 0$):} The KKT conditions give $\int_\Sigma (\tr_\Sigma k) \psi_1 \, dA \ge 0$. The upgrade to pointwise $\tr_\Sigma k \ge 0$ is Conjecture~C (now unconditionally proved for general $k$ in Theorem~\ref{thm:IntegralToPointwise}).
"

lines[127] = new_text
lines[128] = ""

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.writelines(lines)
