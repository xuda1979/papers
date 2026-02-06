import re

file_path = "black_hole_stability_conjecture.tex"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

def find_line(pattern, start_idx=0):
    for i in range(start_idx, len(lines)):
        if pattern in lines[i]:
            return i
    return -1

# 1. Clean Innovations Section
# Start at Innovation 4, Delete until Proof Overview
start_pat_inn = r"\subsection{Innovation 4: The Stability-Complexity Duality}"
end_pat_inn = r"\section{Proof of the Main Theorem: Overview}"
# Actually, the start of the next section is fine.

# 2. Delete Observational Tests
# From "Observational Tests..." to "Scattering Theory"
start_pat_obs = r"\section{Observational Tests of Black Hole Stability}"
end_pat_obs = r"\section{Scattering Theory and Decay Mechanisms}"

# 3. Delete Spectral Quantization
# From "Spectral Quantization..." to "Proof of the Main Theorem" (Note: section number might be different)
# Wait, "Proof of Main Theorem" (2108) is the target.
start_pat_spec = r"\section{Spectral Quantization and Thermodynamic Correspondence}"
end_pat_spec = r"\section{Proof of the Main Theorem}\label{sec:full-proof}"

deletion_markers = [
    (start_pat_inn, end_pat_inn),
    (start_pat_obs, end_pat_obs),
    (start_pat_spec, end_pat_spec)
]

exclusions = []
for start_pat, end_pat in deletion_markers:
    start_idx = find_line(start_pat)
    end_idx = find_line(end_pat)
    
    if start_idx != -1 and end_idx != -1:
        print(f"Found block: {start_pat[:30]}... ({start_idx}) to {end_pat[:30]}... ({end_idx})")
        exclusions.append((start_idx, end_idx))
    else:
        print(f"WARNING: Could not find block {start_pat} -> {end_pat}")

# Sort exclusions
exclusions.sort()

# Build new content
new_lines = []
current_idx = 0

for start, end in exclusions:
    new_lines.extend(lines[current_idx:start])
    current_idx = end

new_lines.extend(lines[current_idx:])

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Innovations and sections cleaned.")
