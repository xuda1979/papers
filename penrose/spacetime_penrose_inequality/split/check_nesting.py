import re

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

depth = 0
for i, line in enumerate(text.splitlines()):
    if "\begin{itemize}" in line:
        depth += 1
        print(f"Line {i+1}: begin itemize, depth={depth}")
    if "\end{itemize}" in line:
        depth -= 1
        print(f"Line {i+1}: end itemize, depth={depth}")
