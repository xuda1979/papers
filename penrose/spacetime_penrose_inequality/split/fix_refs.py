import glob

replacements = [
    (r"Theorem~ef{thm:MaxAreaTrapped}", "our earlier variational approach"),
    (r"Lemma~ef{lem:VanishingMultiplier}", "the earlier vanishing multiplier hypothesis"),
    (r"Conjecture~ef{conj:IntegralToPointwise}", "the localized multiplier realization problem (Open Problem~\ref{op:LocalizedMultiplierRealization})"),
    (r"Theorem~ef{thm:IntegralToPointwise}", "Theorem~\ref{thm:IntegralToPointwiseAppendix}"),
]

files = glob.glob('*.tex')

for filename in files:
    with open(filename, 'r') as f:
        content = f.read()
    
    original_content = content
    for old, new in replacements:
        content = content.replace(old, new)
        
    if content != original_content:
        print(f"Updated {filename}")
        with open(filename, 'w') as f:
            f.write(content)
