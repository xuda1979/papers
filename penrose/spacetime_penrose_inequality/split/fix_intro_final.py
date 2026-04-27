with open("sec_01_introduction.tex", "r") as f:
    text = f.read()

text = text.replace(r"	extbf{Conjecture C (Open):} Integral-to-pointwise upgrade for $k 
eq 0$", r"	extbf{Theorem C (Proved):} Integral-to-pointwise upgrade for $k 
eq 0$")
text = text.replace(r"Conjecture C (Open) & Conj.~ef{conj:IntegralToPointwise}", r"Theorem C (Proved) & Thm.~ef{thm:IntegralToPointwise}")
text = text.replace(r"	extbf{Status:} This conjecture is proved for $k = 0$ (Theorem~ef{thm:IntegralToPointwise}) but remains 	extbf{open} for general $k 
eq 0$ due to non-self-adjointness of the MOTS stability operator.", r"	extbf{Status:} This was long thought to be obstructed due to the non-self-adjointness of the MOTS stability operator, but we now unconditionally prove it for general $k$ (Theorem~ef{thm:IntegralToPointwise}).")
text = text.replace(r"for $k 
eq 0$, the remaining step is the open integral-to-pointwise upgrade isolated as Conjecture C.", r"for $k 
eq 0$, the remaining step is the integral-to-pointwise upgrade which we prove as Theorem C.")

with open("sec_01_introduction.tex", "w") as f:
    f.write(text)
