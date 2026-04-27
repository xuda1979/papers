import os

with open('sec_34_logical_structure_and_gap_closure.tex', 'r') as f:
    content = f.read()

content = content.replace(r'Theorem~ef{thm:MaxAreaTrapped}', 'our earlier variational approach')

with open('sec_34_logical_structure_and_gap_closure.tex', 'w') as f:
    f.write(content)
