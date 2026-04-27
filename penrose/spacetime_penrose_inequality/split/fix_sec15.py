import os

path = 'sec_15_conclusion_and_outlook.tex'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Handle the specific broken block
    if 'Variational-to-Pointwise Favorable Jump' in line:
        if 'begin{openproblem}' in line:
            line = line.replace('begin{openproblem}', 'begin{proposition}')
        elif 'begin{proposition}' in line:
            pass # already fixed
    
    if '\end{openproblem}' in line:
        line = line.replace('\end{openproblem}', '\end{proposition}')
        
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
