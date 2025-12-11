import re

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    orig_path = 'angular_momentum_penrose_theorem.tex'
    cmp_path = 'angular_momentum_penrose_theorem_CMP.tex'

    orig_content = read_file(orig_path)
    cmp_content = read_file(cmp_path)

    # 1. Extract Body and Bibliography from Original
    # Find start of Introduction
    intro_match = re.search(r'\\section\{Introduction\}', orig_content)
    if not intro_match:
        print("Could not find \\section{Introduction} in original file")
        return
    
    # Find start of Bibliography
    bib_match = re.search(r'\\begin\{thebibliography\}', orig_content)
    if not bib_match:
        print("Could not find \\begin{thebibliography} in original file")
        return

    body_start = intro_match.start()
    body_end = bib_match.start()
    
    body_content = orig_content[body_start:body_end]
    
    # Extract Bibliography
    bib_end_match = re.search(r'\\end\{document\}', orig_content)
    if not bib_end_match:
        print("Could not find \\end{document} in original file")
        return
        
    bib_content = orig_content[bib_match.start():bib_end_match.start()]

    # 2. Extract Acknowledgments from Original \thanks
    thanks_match = re.search(r'\\thanks\{(.*?)\}', orig_content, re.DOTALL)
    acknowledgments_text = ""
    if thanks_match:
        acknowledgments_text = thanks_match.group(1).replace('\n', ' ').strip()
        # Clean up multiple spaces
        acknowledgments_text = re.sub(r'\s+', ' ', acknowledgments_text)
    else:
        acknowledgments_text = "The author is grateful to the anonymous referees for their helpful comments."

    # 3. Prepare CMP Preamble
    # We want everything up to \tableofcontents and the following \newpage
    # In the CMP file, it looks like:
    # \tableofcontents
    #
    # \newpage
    #
    # %=============================================================================
    # % MAIN TEXT
    
    toc_match = re.search(r'\\tableofcontents', cmp_content)
    if not toc_match:
        print("Could not find \\tableofcontents in CMP file")
        return

    # Find the \newpage after TOC
    newpage_match = re.search(r'\\newpage', cmp_content[toc_match.end():])
    if not newpage_match:
        print("Could not find \\newpage after TOC in CMP file")
        return
    
    preamble_end = toc_match.end() + newpage_match.end()
    cmp_preamble = cmp_content[:preamble_end]

    # 4. Construct New Content
    new_content = cmp_preamble + "\n\n"
    new_content += "%=============================================================================\n"
    new_content += "% MAIN TEXT (Imported from original manuscript)\n"
    new_content += "%=============================================================================\n\n"
    new_content += body_content
    
    new_content += "\n\n"
    new_content += "%=============================================================================\n"
    new_content += "% ACKNOWLEDGMENTS & STATEMENTS\n"
    new_content += "%=============================================================================\n\n"
    
    new_content += "\\vspace{1cm}\n"
    new_content += "\\noindent\n"
    new_content += "\\textbf{Acknowledgments.} " + acknowledgments_text + "\n\n"
    
    new_content += "\\vspace{0.5cm}\n"
    new_content += "\\noindent\n"
    new_content += "\\textbf{Data Availability Statement.} This manuscript has no associated data, as it is a theoretical mathematical physics paper containing only analytical results.\n\n"
    
    new_content += "\\vspace{0.5cm}\n"
    new_content += "\\noindent\n"
    new_content += "\\textbf{Conflict of Interest Statement.} The author declares no conflicts of interest.\n\n"
    
    new_content += "\\newpage\n\n"
    new_content += bib_content
    new_content += "\n\\end{document}\n"

    # 5. Write to CMP file
    write_file(cmp_path, new_content)
    print("Successfully merged content into " + cmp_path)

if __name__ == "__main__":
    main()
