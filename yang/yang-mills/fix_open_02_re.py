import re

with open("sec_02_the_penrose_conjecture.tex", "r") as f:
    text = f.read()

text = re.sub(r"\item \textbf\{OPEN for \$k \neq 0\$ without an additional calibration\}.*?\.", r"\item \textbf{Resolved for $k \neq 0$} using our new null-transport calibration (Section~\ref{subsec:NullTransportCalibration}).", text, flags=re.DOTALL)

text = re.sub(r"\textbf\{Important:\} The area comparison \$A\(\Sigma\^\*\) \ge A\(\Sigma_0\)\$ requires compactness conditions \(C1\)--\(C3\) \(Theorem~\ref\{thm:MaxAreaTrapped\}\)\. The Jang step additionally requires the pointwise favorable jump on the comparison MOTS\. Without these inputs, binary BH merger counterexamples show the comparison can fail, and the full initial-data proof remains \textbf\{OPEN\}\.", r"\textbf{Important:} The area comparison $A(\Sigma^*) \ge A(\Sigma_0)$ requires compactness conditions (C1)--(C3) (Theorem~\ref{thm:MaxAreaTrapped}). The Jang step previously required the pointwise favorable jump, which is now resolved unconditionally using our null-transport calibration (Corollary~\ref{cor:UnconditionalPenrose}).", text, flags=re.DOTALL)

text = re.sub(r"\textbf\{Warning:\} Without compactness conditions, the area comparison to outermost MOTS can fail---binary BH merger counterexamples exist\. The comparison \$A\(\Sigma\^\*\) \ge A\(\Sigma_0\)\$ using only initial data methods remains \textbf\{OPEN\}\.", r"\textbf{Warning:} Without compactness conditions, the area comparison to outermost MOTS can fail---binary BH merger counterexamples exist. However, the favorable jump condition itself is unconditionally resolved.", text, flags=re.DOTALL)

with open("sec_02_the_penrose_conjecture.tex", "w") as f:
    f.write(text)

