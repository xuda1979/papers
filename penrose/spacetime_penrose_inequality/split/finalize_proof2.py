import os

def final_repair(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        line = line.replace('\b\b', '\')
        line = line.replace('\t\t', '\')
        line = line.replace('\b\begin', '\begin')
        line = line.replace('\t\textbf', '\textbf')
        line = line.replace('\t\tr', '\tr')
        line = line.replace('\t\\', '\')
        line = line.replace('\b\bar', '\bar')
        
        # Address split strings
        line = line.replace('egin{', '\begin{')
        line = line.replace('extbf{', '\textbf{')
        line = line.replace('ef{', '\ref{')
        line = line.replace('ar{', '\bar{')
        
        # Deduplicate
        line = line.replace('\\\', '\')
        line = line.replace('\\begin', '\begin')
        line = line.replace('\\textbf', '\textbf')
        line = line.replace('\\ref', '\ref')
        line = line.replace('\\tr', '\tr')
        line = line.replace('\\bar', '\bar')
        
        new_lines.append(line)
        
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

files = [
    'sec_01_introduction.tex',
    'sec_34_logical_structure_and_gap_closure.tex'
]

for f in files:
    if os.path.exists(f):
        final_repair(f)
