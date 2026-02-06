import os

file_path = "black_hole_stability_conjecture.tex"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

def find_line(pattern, start_idx=0):
    for i in range(start_idx, len(lines)):
        if pattern in lines[i]:
            return i
    return -1

# Define deletion blocks (Start Pattern, End Pattern)
# We will identify indexes and then build a "keep list"
deletion_markers = [
    (r"\section{Higher Dimensions and String Theory Connections}", r"\section{The Master Stability Theorem}"),
    (r"\section{Observational Tests and LIGO/Virgo Connections}", r"\section{Rigorous Proof of Teukolsky-Starobinsky Coercivity}"),
    (r"\section{Entropic Stability: The Second Law Mechanism}", r"\section{The Centerpiece Theorem: Coercivity Implies Spectral Gap}"),
    (r"\section{Grand Stability Correspondence: Rigorous Proof for Schwarzschild}", r"\section{Novel QNM Frequency Inequality}"),
    (r"\section{Neural Network Discovery of Stability Multipliers}", r"\section{Conclusion}"),
    # Note: Conclusion happens, then Promotional stuff follows.
    (r"\section{Tensor Network Models and Emergent Spacetime}", r"\section{Notation and Conventions}")
]

# Calculate exclusion ranges
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
    # Append content before the block
    new_lines.extend(lines[current_idx:start])
    # Skip the block (move current_idx to end)
    current_idx = end

# Append remaining content
new_lines.extend(lines[current_idx:])

# Remove \input{numerical_simulation_section}
final_lines = []
for line in new_lines:
    if r"\input{numerical_simulation_section}" in line:
        continue
    final_lines.append(line)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(final_lines)

print("File pruned successfully.")
