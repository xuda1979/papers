file_path = "black_hole_stability_conjecture.tex"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

try:
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if r"\section{Conclusion}" in line:
            start_idx = i
        if r"\section{Notation and Conventions}" in line:
            end_idx = i
            break # Stop at first occurrence
            
    if start_idx != -1 and end_idx != -1:
        new_conclusion = [
            "\\section{Conclusion}\n",
            "\n",
            "In this work, we have provided a conditional roadmap for the nonlinear stability of the Kerr black hole family in the full subextremal range $|a| < M$. The manuscript does not prove the conjecture; it identifies the analytic estimates that would be needed for such a proof and shows how they would combine if supplied.\n",
            "\n",
            "The remaining proof obligations are the construction of a coercive full-subextremal energy, a Morawetz estimate controlling trapping and superradiance, and a nonlinear gauge/modulation framework compatible with the weak-null hierarchy.\n",
            "\n",
            "\\newpage\n",
            "\n"
        ]
        
        final_lines = lines[:start_idx] + new_conclusion + lines[end_idx:]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)
            
        print("Conclusion rewritten successfully.")
    else:
        print(f"Could not find start/end indices. Start: {start_idx}, End: {end_idx}")

except Exception as e:
    print(f"Error: {e}")
