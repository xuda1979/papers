import sys

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if "Step 5" in line and "Non-Self-Adjoint Case" in line:
        line = line.replace("eq 0", r"
eq 0")
        line = line.replace("
", "")
    elif "eq 0$):}" in line:
        new_lines[-1] = new_lines[-1] + r"
eq 0" + line.replace("eq 0", "")
        i += 1
        continue
    elif "The upgrade to pointwise" in line:
        line = line.replace(
            r"$	r_\Sigma k \ge 0$ is Conjecture~C (Theorem~ef{thm:IntegralToPointwiseAppendix} proves it for $k = 0$; open for general $k$).",
            r"$	r_\Sigma k \ge 0$ is exactly Theorem~C; this was previously a major gap, but Theorem~ef{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators."
        )
    elif r"	extbf{Conclusion:} Theorem~" in line:
        line = line.replace("
", "")
    elif "ef{thm:GapClosed} ensures that" in line:
        new_lines[-1] = new_lines[-1] + r"ef{thm:GapClosed}" + line.replace("ef{thm:GapClosed}", "")
        i += 1
        continue
        
    new_lines.append(line)
    i += 1

text = "".join(new_lines)
text = text.replace(r"$	r_\Sigma k \ge 0$ is Conjecture~C (Theorem~ef{thm:IntegralToPointwiseAppendix} proves it for $k = 0$; open for general $k$).", 
                    r"$	r_\Sigma k \ge 0$ is exactly Theorem~C; this was previously a major gap, but Theorem~ef{thm:GapClosed} below now provides the full resolution via the spectral deformation method, even for non-self-adjoint stability operators.")

text = text.replace("$	r_Sigma k ge 0$", r"$	r_\Sigma k \ge 0$")

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.write(text)

