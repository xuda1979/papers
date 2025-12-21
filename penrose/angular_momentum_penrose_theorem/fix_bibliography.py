"""
Fix the split files to handle the embedded bibliography.
"""

# Read app04-subext-improvement.tex
with open('split/app04-subext-improvement.tex', 'r', encoding='utf-8') as f:
    app04_content = f.read()

# Split into before/during/after bibliography
lines = app04_content.split('\n')

# Find the bibliography markers
bib_start = None
bib_end = None
for i, line in enumerate(lines):
    if '\\begin{thebibliography}' in line:
        bib_start = i
    if '\\end{thebibliography}' in line:
        bib_end = i
        break

if bib_start and bib_end:
    # Extract the actual appendix content (before bibliography)
    appendix_content = '\n'.join(lines[:bib_start-6])  # -6 to remove the REFERENCES comment block too
    
    # Extract the bibliography
    bibliography_content = '\n'.join(lines[bib_start:bib_end+1])
    
    # Check if there's content after the bibliography (for app05)
    # Lines 423+ is the app05 content (Mars-Simon)
    
    # Write the fixed app04 file
    with open('split/app04-subext-improvement.tex', 'w', encoding='utf-8') as f:
        f.write(appendix_content)
    print(f"Fixed app04-subext-improvement.tex: {len(appendix_content.split(chr(10)))} lines")
    
    # Write the bibliography as a separate file
    with open('split/bibliography.tex', 'w', encoding='utf-8') as f:
        f.write(bibliography_content)
    print(f"Created bibliography.tex: {len(bibliography_content.split(chr(10)))} lines")
else:
    print("Could not find bibliography markers")

# Now we need to update main.tex to use the embedded bibliography instead of bibtex
main_content = open('split/main.tex', 'r', encoding='utf-8').read()

# Replace the bibtex commands with input to the bibliography file
main_content = main_content.replace(
    '\\bibliographystyle{plain}\n\\bibliography{references}',
    '\\input{bibliography}'
)

with open('split/main.tex', 'w', encoding='utf-8') as f:
    f.write(main_content)
print("Updated main.tex to use embedded bibliography")

print("\nDone!")
