"""
Script to extract sections from the CMP file into separate split files.
"""
import re

# Read the CMP file
with open('angular_momentum_penrose_theorem_CMP.tex', 'r', encoding='utf-8') as f:
    cmp_lines = f.readlines()

# Define section mappings (start_line is 1-indexed, end_line is inclusive)
# Based on grep_search results for section markers
sections = {
    'sec04-jang.tex': (946, 1904),
    'sec05-lichnerowicz.tex': (1905, 2735),
    'sec06-amo.tex': (2736, 4178),
    'sec07-subextremality.tex': (4179, 4288),
    'sec08-synthesis.tex': (4289, 4551),
    'sec09-rigidity.tex': (4552, 4930),
    'sec10-extensions.tex': (4931, 5794),
}

# Appendices (based on grep for \section{...}\label{app:...})
appendices = {
    'app01-amo-estimates.tex': (6089, 6181),
    'app02-schauder.tex': (6182, 6342),
    'app03-supersolution.tex': (6343, 6441),
    'app04-subext-improvement.tex': (6442, 6867),
    'app05-mars-simon.tex': (6868, 7085),  # Until end of file
}

# Also need sec11-numerical, sec12-technical, sec13-conclusion
# These correspond to the numerical appendix and conclusion in CMP
additional_sections = {
    'sec11-numerical.tex': (5795, 5888),  # Actually Appendix A: Numerical Illustrations
    'sec12-technical.tex': (5889, 5985),  # Actually Appendix B: Technical Foundations
    'sec13-conclusion.tex': (5986, 6088),  # Section: Conclusion
}

def extract_section(start_line, end_line):
    """Extract lines from CMP file (1-indexed, inclusive)"""
    # Convert to 0-indexed
    return ''.join(cmp_lines[start_line-1:end_line])

# Extract and write main sections
print("Extracting main sections...")
for filename, (start, end) in sections.items():
    content = extract_section(start, end)
    filepath = f'split/{filename}'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Written {filepath}: lines {start}-{end} ({end-start+1} lines)")

# Extract and write additional sections
print("\nExtracting additional sections...")
for filename, (start, end) in additional_sections.items():
    content = extract_section(start, end)
    filepath = f'split/{filename}'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Written {filepath}: lines {start}-{end} ({end-start+1} lines)")

# Extract and write appendices
print("\nExtracting appendices...")
for filename, (start, end) in appendices.items():
    content = extract_section(start, end)
    filepath = f'split/{filename}'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Written {filepath}: lines {start}-{end} ({end-start+1} lines)")

print("\nDone! All sections extracted.")
