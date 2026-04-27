import sys

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    lines = f.readlines()

new_lines = []
skip_next = False
for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue

    if "Step 5 (Non-Self-Adjoint Case, $k" in line:
        new_lines.append(r"	extbf{Step 5 (Non-Self-Adjoint Case, $k 
eq 0$):} The KKT conditions give $\int_\Sigma (	r_\Sigma k) \psi_1 \, dA \ge 0$. The upgrade to pointwise $	r_\Sigma k \ge 0$ is exactly Theorem~C; this was previously a major gap, but Theorem~ef{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators." + "
")
        skip_next = True # Skip the next line which has the 'eq 0...' part
        continue

    if "Conclusion:" in line:
        new_lines.append(r"	extbf{Conclusion:} Theorem~ef{thm:GapClosed} ensures that the favorable jump condition $	r_\Sigma k \ge 0$ depends only on initial data geometry, not on any property of the Jang metric. The Jang equation is solved \emph{after} $\Sigma$ is selected, and its blow-up behavior \emph{follows from} (rather than \emph{determines}) the trapped surface geometry." + "
")
        skip_next = True # Skip the next line which has the 'ef{thm...' part
        continue
        
    new_lines.append(line)

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.writelines(new_lines)

