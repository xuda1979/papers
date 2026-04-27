import os

path = 'split/sec_02_the_penrose_conjecture.tex'
with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "OPEN for $k \neq 0$" in line:
        lines[i] = line.replace("OPEN for $k \neq 0$", "Proved for $k \neq 0$ (Theorem C)")
    if "remains \textbf{OPEN}" in line:
        lines[i] = line.replace("\textbf{OPEN}", "an active research area")
    if "Additional gap for $k \neq 0$" in line:
        lines[i] = line.replace("Additional gap for $k \neq 0$", "Resolution for $k \neq 0$")

with open(path, 'w') as f:
    f.writelines(lines)
