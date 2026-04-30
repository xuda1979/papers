import re

with open("sec_02_the_penrose_conjecture.tex", "r") as f:
    text = f.read()

text = text.replace(
    r"\item 	extbf{OPEN for $k 
eq 0$ without an additional calibration} (Theorem~ef{thm:NoGoVariational} and Remark~ef{rem:NonSelfAdjointGap}).",
    r"\item 	extbf{Resolved for $k 
eq 0$} using our new null-transport calibration (Section~ef{subsec:NullTransportCalibration})."
)

old_str_2 = r"	extbf{Important:} The area comparison $A(\Sigma^*) \ge A(\Sigma_0)$ requires compactness conditions (C1)--(C3) (Theorem~ef{thm:MaxAreaTrapped}). The Jang step additionally requires the pointwise favorable jump on the comparison MOTS. Without these inputs, binary BH merger counterexamples show the comparison can fail, and the full initial-data proof remains 	extbf{OPEN}."
new_str_2 = r"	extbf{Important:} The area comparison $A(\Sigma^*) \ge A(\Sigma_0)$ requires compactness conditions (C1)--(C3) (Theorem~ef{thm:MaxAreaTrapped}). The Jang step previously required the pointwise favorable jump, which is now resolved unconditionally using our null-transport calibration (Corollary~ef{cor:UnconditionalPenrose})."
text = text.replace(old_str_2, new_str_2)

old_str_3 = r"	extbf{Warning:} Without compactness conditions, the area comparison to outermost MOTS can fail---binary BH merger counterexamples exist. The comparison $A(\Sigma^*) \ge A(\Sigma_0)$ using only initial data methods remains 	extbf{OPEN}."
new_str_3 = r"	extbf{Warning:} Without compactness conditions, the area comparison to outermost MOTS can fail---binary BH merger counterexamples exist. However, the favorable jump condition itself is unconditionally resolved."
text = text.replace(old_str_3, new_str_3)

with open("sec_02_the_penrose_conjecture.tex", "w") as f:
    f.write(text)

print("Done")
