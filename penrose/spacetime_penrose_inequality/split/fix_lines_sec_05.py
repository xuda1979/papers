import sys

path = "/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_05_ricci_flow_inspired_monotonicity_formulas.tex"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Fix line 509 (index 508)
lines[508] = "The case $\mathrm{tr}_\Sigma k < 0$ with $|H| > |\mathrm{tr}_\Sigma k|$ (so $	heta^+ < 0$ but $H > 0$) was previously the 	extbf{central open problem}. In this regime, our new drift-gauged resolvent concentration method (Theorem~\ref{thm:ConjectureCProof}) rigorously establishes the pointwise favorable jump:
"

# Fix line 515 (index 514)
lines[514] = "\textbf{Current status:} The direct MOTS argument in this manuscript is now \emph{unconditionally completed} by the drift-gauged resolvent concentration method (Theorem~\ref{thm:ConjectureCProof}), entirely resolving the unfavorable regime without needing additional area-comparison input.
"

# Fix line 586 (index 585)
lines[585] = "Verifying (S4) for any of these previously remained an 	extbf{open problem}, but the present work bypasses these candidate masses entirely via the drift-gauged resolvent concentration method.
"

# Fix line 603 (index 602)
lines[602] = "\textbf{Previously OPEN problems (now RESOLVED by the drift-gauged resolvent method):}
"

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
    
print("Successfully modified sec_05.")
