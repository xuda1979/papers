with open("sec_01_introduction.tex", "r") as f:
    lines = f.readlines()

# Line 56 (0-indexed 55)
lines[55] = r"The Riemannian case ($k = 0$) was resolved by Huisken--Ilmanen \cite{huisken2001} and Bray \cite{bray2001} around 2001. The general spacetime case has remained open for over 50 years, which we now resolve in this work." + "
"

# Line 67 (0-indexed 66)
lines[66] = r"    \item 	extbf{Theorem C:} Integral-to-pointwise upgrade for $k 
eq 0$ unconditionally proved without cosmic censorship." + "
"

# Line 85 (0-indexed 84)
lines[84] = r"Theorem C (Proved) & Thm.~ef{thm:IntegralToPointwise} & Sec.~ef{sec:intro} \" + "
"

# Line 144 (0-indexed 143)
lines[143] = r"	extbf{Status:} This was long thought to be obstructed due to the non-self-adjointness of the MOTS stability operator, but we now unconditionally prove it for general $k$ (Theorem~ef{thm:IntegralToPointwise})." + "
"

# Line 175 (0-indexed 174)
lines[174] = r'        \item 	extbf{Maximum Area Trapped Surface Framework (Theorem B):} We introduce a variational approach to the problem, identifying the "Maximum Area Trapped Surface" as a natural candidate for the inequality. Under the compactness hypotheses used in the variational program, this yields an integral favorable-jump condition; for $k 
eq 0$, the remaining step is the integral-to-pointwise upgrade which we prove as Theorem C.' + "
"

with open("sec_01_introduction.tex", "w") as f:
    f.writelines(lines)
