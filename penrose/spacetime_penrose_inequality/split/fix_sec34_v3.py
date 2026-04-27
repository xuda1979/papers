with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

# Fix Step 5
if "	extbf{Step 5 (Non-Self-Adjoint Case, $k 
" in text:
    old_s = "	extbf{Step 5 (Non-Self-Adjoint Case, $k 
eq 0$):} The KKT conditions give $\int_\Sigma (	r_\Sigma k) \psi_1 \, dA \ge 0$. The upgrade to pointwise $	r_\Sigma k \ge 0$ is Conjecture~C (Theorem~ef{thm:IntegralToPointwiseAppendix} proves it for $k = 0$; open for general $k$)."
    new_s = "\textbf{Step 5 (Non-Self-Adjoint Case, $k \neq 0$):} The KKT conditions give $\int_\Sigma (\tr_\Sigma k) \psi_1 \, dA \ge 0$. The upgrade to pointwise $\tr_\Sigma k \ge 0$ is exactly Theorem~C; this was previously a major gap, but Theorem~\ref{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators."
    text = text.replace(old_s, new_s)
    
# Try to catch the broken string case
import re
text = re.sub(
    r"\textbf\{Step 5 \(Non-Self-Adjoint Case, \$k 
(.*?)general \$k\)\.
?",
    r"\textbf{Step 5 (Non-Self-Adjoint Case, $k \neq 0$):} The KKT conditions give $\int_\Sigma (\tr_\Sigma k) \psi_1 \, dA \ge 0$. The upgrade to pointwise $\tr_\Sigma k \ge 0$ is exactly Theorem~C; this was previously a major gap, but Theorem~\ref{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators.
",
    text,
    flags=re.DOTALL
)

# Fix Conclusion
text = re.sub(
    r"\textbf\{Conclusion:\} Theorem~
(.*?)ge 0\$",
    r"\textbf{Conclusion:} Theorem~\ref{thm:GapClosed} ensures that the favorable jump condition $\tr_\Sigma k \ge 0$",
    text,
    flags=re.DOTALL
)

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.write(text)
