import os

path = 'sec_15_conclusion_and_outlook.tex'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
in_openproblem = False
for line in lines:
    if 'begin{openproblem}' in line:
        in_openproblem = True
    
    if in_openproblem and 'end{proposition}' in line:
        line = line.replace('end{proposition}', 'end{openproblem}')
        in_openproblem = False
    
    if in_openproblem and 'end{openproblem}' in line:
        in_openproblem = False
        
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
