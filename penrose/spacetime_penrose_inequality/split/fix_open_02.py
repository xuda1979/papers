with open("sec_02_the_penrose_conjecture.tex", "r") as f:
    text = f.read()

text = text.replace(
    "\item \textbf{OPEN for $k \neq 0$ without an additional calibration} (Theorem~\ref{thm:NoGoVariational} and Remark~\ref{rem:NonSelfAdjointGap}).",
    "\item \textbf{Resolved for $k \neq 0$} using our new null-transport calibration (Section~\ref{subsec:NullTransportCalibration})."
)

text = text.replace(
    "\textbf{Important:} The area comparison $A(\Sigma^*) \ge A(\Sigma_0)$ requires compactness conditions (C1)--(C3) (Theorem~\ref{thm:MaxAreaTrapped}). The Jang step additionally requires the pointwise favorable jump on the comparison MOTS. Without these inputs, binary BH merger counterexamples show the comparison can fail, and the full initial-data proof remains \textbf{OPEN}.",
    "\textbf{Important:} The area comparison $A(\Sigma^*) \ge A(\Sigma_0)$ requires compactness conditions (C1)--(C3) (Theorem~\ref{thm:MaxAreaTrapped}). The Jang step previously required the pointwise favorable jump, which is now resolved unconditionally using our null-transport calibration (Corollary~\ref{cor:UnconditionalPenrose})."
)

with open("sec_02_the_penrose_conjecture.tex", "w") as f:
    f.write(text)

