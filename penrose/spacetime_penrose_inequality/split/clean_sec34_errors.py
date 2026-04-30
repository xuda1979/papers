import re

with open('/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_34_logical_structure_and_gap_closure.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace any stray backspace characters \x08
content = content.replace("\x08", "")

# Fix missing backslashes caused by python raw strings misinterpretation
content = content.replace(" egin{theorem}", "\begin{theorem}")
content = content.replace(" egin{proof}", "\begin{proof}")
content = content.replace("	heta^+", "\theta^+")
content = content.replace("	r_\Sigma", "\tr_\Sigma")
content = content.replace("	ext{", "\text{")
content = content.replace("+ho", "\rho")
content = content.replace("+ef", "\ref")
content = content.replace("+abla", "\nabla")
content = content.replace("+angle", "\rangle")

with open('/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_34_logical_structure_and_gap_closure.tex', 'w', encoding='utf-8') as f:
    f.write(content)
