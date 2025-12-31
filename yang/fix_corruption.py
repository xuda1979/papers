import sys
import os

filepath = r'c:\Users\Lenovo\papers\yang\yang_mills\final_proof\sec_roadmap1_lattice_gap.tex'

if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
    sys.exit(1)

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    stripped = line.rstrip('\n')
    if len(stripped) > 0 and len(stripped) % 2 == 0:
        mid = len(stripped) // 2
        if stripped[:mid] == stripped[mid:]:
            new_lines.append(stripped[:mid] + '\n')
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("File fixed.")
