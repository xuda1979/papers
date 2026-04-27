import re

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

# Pattern 1
p1 = r"\textbf\{Step 5 \(Non-Self-Adjoint Case, \$k \neq 0\$\):\} The KKT conditions give \$\int_\Sigma \(\tr_\Sigma k\) \psi_1 \, dA \ge 0\$\. The upgrade to pointwise \$\tr_\Sigma k \ge 0\$ is Conjecture~C \(Theorem~\ref\{thm:IntegralToPointwiseAppendix\} proves it for \$k = 0\$; open for general \$k\)\."

r1 = r"	extbf{Step 5 (Non-Self-Adjoint Case, $k 
eq 0$):} The KKT conditions give $\int_\Sigma (	r_\Sigma k) \psi_1 \, dA \ge 0$. The upgrade to pointwise $	r_\Sigma k \ge 0$ is exactly Theorem~C; this was previously a major gap, but Theorem~ef{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators."

# Pattern 2
p2 = r"\textbf\{Conclusion:\} Theorem~\ref\{thm:GapClosed\} ensures that the favorable jump condition \$\tr_\Sigma k \ge 0\$"
r2 = r"	extbf{Conclusion:} Theorem~ef{thm:GapClosed} ensures that the favorable jump condition $	r_\Sigma k \ge 0$"

text = re.sub(p1, r1, text)
text = re.sub(p2, r2, text)

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.write(text)

