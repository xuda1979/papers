import glob

files = [
    'sec_01_introduction.tex',
    'sec_02_the_penrose_conjecture.tex',
    'sec_03_overview.tex',
    'sec_04_the_theta_plus_flow_method.tex',
    'sec_05_ricci_flow_inspired_monotonicity_formulas.tex',
    'sec_10_synthesis_limit_of_inequalities.tex',
    'sec_12_complete_rigorous_proof_consolidated_statement.tex',
    'sec_13_index_of_notation.tex'
]

for filename in files:
    with open(filename, 'r') as f:
        content = f.read()
    
    # replace literal strings
    content = content.replace("Theorem~\ref{thm:MaxAreaTrapped}", "the variational MOTS approach")
    
    with open(filename, 'w') as f:
        f.write(content)
