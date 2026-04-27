import sys
with open('sec_34_logical_structure_and_gap_closure.tex', 'r') as f:
    t = f.read()
t = t.replace(' egin{theorem}', '\begin{theorem}')
with open('sec_34_logical_structure_and_gap_closure.tex', 'w') as f:
    f.write(t)
