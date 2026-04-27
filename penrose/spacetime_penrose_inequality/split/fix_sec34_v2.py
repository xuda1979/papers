import sys

# We use double backslashes in the python string, 
# but cat << 'EOF' prevents the shell from eating them.

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    lines = f.readlines()

new_lines = []
skip = 0
for i, line in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue

    # Identify the Step 5 block start
    if "\textbf{Step 5 (Non-Self-Adjoint Case, $k" in line:
        new_lines.append("\textbf{Step 5 (Non-Self-Adjoint Case, $k \neq 0$):} The KKT conditions give $\int_\Sigma (\tr_\Sigma k) \psi_1 \, dA \ge 0$. The upgrade to pointwise $\tr_\Sigma k \ge 0$ is exactly Theorem~C; this was previously a major gap, but Theorem~\ref{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators.
")
        # Check if the next line is the broken fragment "eq 0$):}..."
        if i+1 < len(lines) and "eq 0" in lines[i+1]:
            skip = 1
        continue

    # Identify the Conclusion block start
    if "\textbf{Conclusion:}" in line:
        new_lines.append("\textbf{Conclusion:} Theorem~\ref{thm:GapClosed} ensures that the favorable jump condition $\tr_\Sigma k \ge 0$ depends only on initial data geometry, not on any property of the Jang metric. The Jang equation is solved \emph{after} $\Sigma$ is selected, and its blow-up behavior \emph{follows from} (rather than \emph{determines}) the trapped surface geometry.
")
        # Check if the next line is the broken fragment "ef{thm:GapClosed}..."
        if i+1 < len(lines) and "ef{thm:GapClosed}" in lines[i+1]:
            skip = 1
        continue

    # Fix other common corruption in this file
    line = line.replace("r_Sigma", "\tr_\Sigma")
    line = line.replace("ge 0", "\ge 0")
    line = line.replace("ef{thm:", "\ref{thm:")
    
    new_lines.append(line)

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.writelines(new_lines)
