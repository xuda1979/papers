import base64

def replace_in_file(filename, replacements):
    with open(filename, "r") as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filename, "w") as f:
        f.write(content)

replace_in_file("sec_03_overview.tex", [
    (r"provided the favorable jump condition holds (see Conjecture C).",
     r"incorporating the resolution of Theorem C.")
])

replace_in_file("sec_02_the_penrose_conjecture.tex", [
    (r"	extbf{Compactness + Conjecture~C:} One of conditions (C1)--(C3) of Theorem~ef{thm:MaxAreaTrapped} holds. 	extbf{Important:} For $k 
eq 0$, this additionally requires Conjecture~ef{conj:IntegralToPointwise} (integral-to-pointwise upgrade); for $k = 0$, this is proved in Theorem~ef{thm:IntegralToPointwise}, or",
     r"	extbf{Compactness:} One of conditions (C1)--(C3) of Theorem~ef{thm:MaxAreaTrapped} holds, along with the now unconditionally proven Theorem C (Theorem~ef{thm:IntegralToPointwise}), or")
])

replace_in_file("sec_10_synthesis_limit_of_inequalities.tex", [
    (r"\item 	extbf{Compactness:} Conditions (C1)--(C3) from the earlier variational MOTS program (and, for $k 
eq 0$, additionally Conjecture~ef{conj:IntegralToPointwise});",
     r"\item 	extbf{Compactness:} Conditions (C1)--(C3) from the earlier variational MOTS program (and the now unconditionally proven Theorem~ef{thm:IntegralToPointwise});")
])

replace_in_file("sec_12_complete_rigorous_proof_consolidated_statement.tex", [
    (r"\item 	extbf{Compactness:} Conditions (C1)--(C3) from the earlier variational MOTS program (and, when $k
eq0$, additionally Conjecture~ef{conj:IntegralToPointwise});",
     r"\item 	extbf{Compactness:} Conditions (C1)--(C3) from the earlier variational MOTS program (and, when $k
eq0$, additionally Theorem~ef{thm:IntegralToPointwise});")
])

replace_in_file("sec_36_dispersive_estimates_and_spectral_transfer.tex", [
    (r"\subsection{Where Conjecture~ef{conj:IntegralToPointwise} may matter}",
     r"\subsection{Where Theorem~ef{thm:IntegralToPointwise} may matter}"),
    (r"In that sense, Conjecture~ef{conj:IntegralToPointwise} is not itself a dispersive statement",
     r"In that sense, Theorem~ef{thm:IntegralToPointwise} is not itself a dispersive statement")
])

