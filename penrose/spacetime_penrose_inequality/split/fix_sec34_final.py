import re

with open('/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_34_logical_structure_and_gap_closure.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the specific missing backslashes that pdflatex complained about
content = content.replace(" egin{theorem}", "egin{theorem}")
content = content.replace(" egin{proof}", "egin{proof}")
content = content.replace(" ef{", "ef{")
content = content.replace("
nabla", "\nabla")

with open('/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_34_logical_structure_and_gap_closure.tex', 'w', encoding='utf-8') as f:
    f.write(content)
