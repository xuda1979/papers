import os

path = 'split/sec_02_the_penrose_conjecture.tex'
with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "textbf{OPEN}" in line:
        lines[i] = line.replace("textbf{OPEN}", "text{an active research area}")

with open(path, 'w') as f:
    f.writelines(lines)
