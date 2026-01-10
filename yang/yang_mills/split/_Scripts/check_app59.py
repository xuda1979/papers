import re

filename = 'app59_complete_resolution_of_remaining_gaps_and_conjectu.tex'

with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

stack = []
for i, line in enumerate(lines):
    line_num = i + 1
    if '\\begin{proof}' in line:
        stack.append(line_num)
        print(f"begin at {line_num}")
    if '\\end{proof}' in line:
        if stack:
            start = stack.pop()
            print(f"end at {line_num} matches {start}")
        else:
            print(f"EXTRA END at {line_num}")
            
if stack:
    print(f"UNCLOSED PROOFS starting at {stack}")
