import os

file_path = r"c:\Users\david\papers\adscft\rigorous-adscft\rigorous-adscft.tex"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_marker = r"\section{Mathematical Preliminaries and Rigorous Definitions}"
end_marker = r"\section{Generalized Holography: Beyond Specific Metrics}"

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if start_marker in line:
        start_idx = i
    if end_marker in line:
        end_idx = i

if start_idx != -1 and end_idx != -1:
    print(f"Found start at {start_idx}, end at {end_idx}")
    # We want to keep lines before start_idx
    # We want to delete from start_idx up to end_idx (exclusive, so we keep end_idx)
    
    new_lines = lines[:start_idx] + lines[end_idx:]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Successfully deleted survey sections.")
else:
    print("Could not find markers.")
    if start_idx == -1: print(f"Start marker not found: {start_marker}")
    if end_idx == -1: print(f"End marker not found: {end_idx}")
