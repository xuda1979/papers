def replace_in_file(filename, replacements):
    with open(filename, "r") as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filename, "w") as f:
        f.write(content)

replace_in_file("sec_10_synthesis_limit_of_inequalities.tex", [
    (r"Theorem}ref{thm:IntegralToPointwise}",
     r"Theorem~ef{thm:IntegralToPointwise}")
])

replace_in_file("sec_12_complete_rigorous_proof_consolidated_statement.tex", [
    (r"Theorem}ref{thm:IntegralToPointwise}",
     r"Theorem~ef{thm:IntegralToPointwise}")
])

replace_in_file("sec_36_dispersive_estimates_and_spectral_transfer.tex", [
    (r"Theorem}ref{thm:IntegralToPointwise}",
     r"Theorem~ef{thm:IntegralToPointwise}")
])

replace_in_file("sec_03_overview.tex", [
    (r"Theorem}ref{thm:penroseinitial}",
     r"Theorem~ef{thm:penroseinitial}")
])

replace_in_file("sec_01_introduction.tex", [
    (r"Theorem}~ef{thm:IntegralToPointwise}",
     r"Theorem~ef{thm:IntegralToPointwise}")
])

