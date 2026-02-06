import re

with open('yang_mills_mass_gap.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix tab+extbf -> \textbf  (tab char = \t)
count1 = text.count('\textbf{')
text = text.replace('\textbf{', '\\textbf{')

# Fix tab+exttt -> \texttt
count2 = text.count('\texttt{')
text = text.replace('\texttt{', '\\texttt{')

# Fix tab+ext{ -> \text{  (broken \text commands)
count3 = text.count('\text{')
text = text.replace('\text{', '\\text{')

with open('yang_mills_mass_gap.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Fixed {count1} tab-corrupted textbf, {count2} texttt, {count3} text occurrences")
