import sys
import re

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

# Instead of re.sub with huge blocks, let's just do targeted string replaces.
# We will use triple quotes to avoid needing to escape single quotes,
# and we will read the file line by line to modify specifically the target strings.

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "The upgrade to pointwise $\tr_\Sigma k \ge 0$ is Conjecture~C" in line:
        lines[i] = "\textbf{Step 5 (Non-Self-Adjoint Case, $k \neq 0$):} The KKT conditions give $\int_\Sigma (\tr_\Sigma k) \psi_1 \, dA \ge 0$. The upgrade to pointwise $\tr_\Sigma k \ge 0$ is exactly Theorem~C; this was previously a major gap, but Theorem~\ref{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators.
"

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.writelines(lines)
