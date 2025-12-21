"""
Re-extract sections from CMP file with correct boundaries.
"""
import re

# Read the CMP file
with open('angular_momentum_penrose_theorem_CMP.tex', 'r', encoding='utf-8') as f:
    cmp_lines = f.readlines()

def extract_section(start_line, end_line):
    """Extract lines from CMP file (1-indexed, inclusive)"""
    return ''.join(cmp_lines[start_line-1:end_line])

# Corrected section mappings based on actual content
sections = {
    # First 3 sections are mostly OK, re-extract just to be sure
    'sec01-introduction.tex': (134, 517),
    'sec02-kerr.tex': (518, 642),
    'sec03-proof-outline.tex': (643, 945),
    'sec04-jang.tex': (946, 1904),
    'sec05-lichnerowicz.tex': (1905, 2735),
    'sec06-amo.tex': (2736, 4178),
    'sec07-subextremality.tex': (4179, 4288),
    'sec08-synthesis.tex': (4289, 4551),
    'sec09-rigidity.tex': (4552, 4930),
    'sec10-extensions.tex': (4931, 5790),  # Ends before \appendix
}

# The appendix sections in CMP 
# Note: sec11 and sec12 are in the appendix in CMP but we treat them as sections
appendix_intro = """% Appendices begin here
\\appendix

"""

additional_sections = {
    # These are appendix sections in CMP
    'sec11-numerical.tex': (5795, 5888),  # Appendix: Numerical Illustrations
    'sec12-technical.tex': (5889, 5985),  # Appendix: Technical Foundations
    'sec13-conclusion.tex': (5986, 6087),  # Section: Conclusion (actually not an appendix!)
}

# Appendix sections (after bibliography in logical order, but we reorganize)
appendices = {
    'app01-amo-estimates.tex': (6089, 6180),
    'app02-schauder.tex': (6182, 6342),
    'app03-supersolution.tex': (6343, 6441),
    'app04-subext-improvement.tex': (6442, 6488),  # Just the content, not acknowledgments
    'app05-mars-simon.tex': (6868, 7085),
}

# Acknowledgments and statements (should come at the very end before bibliography)
acknowledgments_content = extract_section(6489, 6515)

# Bibliography (embedded in CMP)
bibliography_content = extract_section(6517, 6864)

# Extract and write main sections
print("Extracting main sections...")
for filename, (start, end) in sections.items():
    content = extract_section(start, end)
    filepath = f'split/{filename}'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Written {filepath}: lines {start}-{end} ({end-start+1} lines)")

# Extract and write additional sections (which are appendices in CMP)
print("\nExtracting additional sections (appendices in CMP)...")
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

# Write acknowledgments
with open('split/acknowledgments.tex', 'w', encoding='utf-8') as f:
    f.write(acknowledgments_content)
print(f"\nWritten split/acknowledgments.tex: {len(acknowledgments_content.split(chr(10)))} lines")

# Write bibliography
with open('split/bibliography.tex', 'w', encoding='utf-8') as f:
    f.write(bibliography_content)
print(f"Written split/bibliography.tex: {len(bibliography_content.split(chr(10)))} lines")

print("\nDone! All sections extracted with correct boundaries.")
