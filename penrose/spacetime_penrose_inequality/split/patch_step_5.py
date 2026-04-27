import os

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

bs = chr(92)
newline = chr(10)

lines = text.splitlines()
for i, line in enumerate(lines):
    if "is Conjecture~C" in line and "open for general" in line:
        lines[i] = bs + "textbf{Step 5 (Non-Self-Adjoint Case, $k " + bs + "neq 0$):} The KKT conditions give $" + bs + "int_Sigma (" + bs + "tr_Sigma k) " + bs + "psi_1 " + bs + ", dA " + bs + "ge 0$. The upgrade to pointwise $" + bs + "tr_Sigma k " + bs + "ge 0$ is exactly Theorem~C; this was previously a major gap, but Theorem~" + bs + "ref{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators."
    
    if bs + "textbf{Conclusion:} The favorable condition" in line:
        lines[i] = bs + "textbf{Conclusion:} Theorem~" + bs + "ref{thm:GapClosed} ensures that the favorable jump condition $" + bs + "tr_Sigma k " + bs + "ge 0$ depends only on initial data geometry, not on any property of the Jang metric. The Jang equation is solved " + bs + "emph{after} $" + bs + "Sigma$ is selected, and its blow-up behavior " + bs + "emph{follows from} (rather than " + bs + "emph{determines}) the trapped surface geometry."

text = newline.join(lines)

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.write(text)
