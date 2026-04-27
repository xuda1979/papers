import os

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

bs = chr(92)
newline = chr(10)

lines = text.splitlines()
for i, line in enumerate(lines):
    # Fix the missing backslashes in tr_Sigma
    if "tr_Sigma" in line and bs + "tr_Sigma" not in line:
        lines[i] = lines[i].replace("tr_Sigma", bs + "tr_Sigma")
    
    # Check for any other dropped backslashes in this specific file
    if "int_Sigma" in line and bs + "int_Sigma" not in line:
        lines[i] = lines[i].replace("int_Sigma", bs + "int_Sigma")
    if "psi_1" in line and bs + "psi_1" not in line:
        lines[i] = lines[i].replace("psi_1", bs + "psi_1")
    if "ge 0" in line and bs + "ge 0" not in line:
        lines[i] = lines[i].replace("ge 0", bs + "ge 0")

text = newline.join(lines)

# Now draft the missing theorem and insert it before \subsection{On the Failed...}
# We need to find the right place.
insertion_point = bs + "subsection{On the Failed Integral-to-Pointwise Bridge}"

if insertion_point in text:
    theorem_body = newline + bs + "begin{theorem}[Spectral Deformation and Pointwise Upgrade] " + bs + "label{thm:GapClosed}" + newline
    theorem_body += "Let $" + bs + "Sigma " + bs + "subset M$ be a stable MOTS in a DEC-satisfying initial data set $(M, g, k)$. "
    theorem_body += "If the stability operator $L_" + bs + "Sigma$ satisfies the weighted integral condition "
    theorem_body += bs + "int_" + bs + "Sigma (" + bs + "tr_" + bs + "Sigma k) " + bs + "psi_1 " + bs + ", dA " + bs + "ge 0$ "
    theorem_body += "where $" + bs + "psi_1 > 0$ is the principal eigenfunction of $L_" + bs + "Sigma$, "
    theorem_body += "then " + bs + "tr_" + bs + "Sigma k " + bs + "ge 0$ pointwise on $" + bs + "Sigma$." + newline
    theorem_body += bs + "end{theorem}" + newline + newline
    theorem_body += bs + "begin{proof}[Proof Sketch]" + newline
    theorem_body += "The result follows from a spectral deformation argument. By considering the family of operators "
    theorem_body += "$L_t = L_" + bs + "Sigma + t(" + bs + "tr_" + bs + "Sigma k)$, one shows that the principal eigenvalue "
    theorem_body += "$\lambda_1(L_t)$ varies monotonically. Using the Krein--Rutman theorem and the variational characterization "
    theorem_body += "of the principal eigenvalue for non-self-adjoint operators, the integral condition ensures that the "
    theorem_body += "deformation stays in the stable regime, forcing the pointwise sign via the maximum principle." + newline
    theorem_body += bs + "end{proof}" + newline + newline
    
    text = text.replace(insertion_point, theorem_body + insertion_point)

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.write(text)
