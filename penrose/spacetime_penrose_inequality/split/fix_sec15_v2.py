import os

path = 'sec_15_conclusion_and_outlook.tex'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
in_unconditional = False
for line in lines:
    if 'begin{openproblem}[Unconditional' in line:
        in_unconditional = True
    
    if in_unconditional and '\end{proposition}' in line:
        line = line.replace('\end{proposition}', '\end{openproblem}')
        in_unconditional = False
        
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
