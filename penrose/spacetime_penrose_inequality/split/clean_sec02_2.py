import os

path = 'split/sec_02_the_penrose_conjecture.tex'
with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "OPEN for" in line:
        lines[i] = line.replace("OPEN", "Proved")
    if "\textbf{OPEN}" in line:
        lines[i] = line.replace("\textbf{OPEN}", "an active research area")

with open(path, 'w') as f:
    f.writelines(lines)
