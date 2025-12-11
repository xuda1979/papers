
import re

def process_tex():
    # Read ns.tex
    with open('ns.tex', 'r', encoding='utf-8') as f:
        ns_lines = f.readlines()

    # Read ns_revised.tex
    with open('ns_revised.tex', 'r', encoding='utf-8') as f:
        revised_lines = f.readlines()

    # Extract content from ns_revised.tex (lines 130-543, 0-indexed is 129-543)
    # The user said lines 130 to 543.
    # Let's verify the content.
    # Line 130 is "\section{Preliminaries}"
    # Line 543 is the end of the proof/section.
    
    new_content_lines = revised_lines[129:543]
    new_content = "".join(new_content_lines)

    # Adjust section levels
    # \section -> \subsection
    # \subsection -> \subsubsection
    # We must do subsection first to avoid double replacement if we used simple replace
    # But regex is safer.
    
    # Replace \subsection with \subsubsection
    new_content = re.sub(r'\\subsection\{', r'\\subsubsection{', new_content)
    # Replace \section with \subsection
    new_content = re.sub(r'\\section\{', r'\\subsection{', new_content)

    # Add label for hyper regularity
    # Find \label{thm:main} and replace with \label{thm:hyper_regularity}
    if 'label{thm:main}' in new_content:
        new_content = new_content.replace('label{thm:main}', 'label{thm:hyper_regularity}')
    else:
        # If not found, append it to the first theorem or just add it
        pass

    # Identify the block to replace in ns.tex
    # Start: "In this section, we present an initial analysis..." (Line 1200 approx)
    # End: "...difficult for small $\alpha$." (Line 1296 approx)
    
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(ns_lines):
        if "In this section, we present an initial analysis" in line:
            start_idx = i
        if "But proving regularity even with" in line and "difficult for small" in line:
            end_idx = i
            break
            
    if start_idx != -1 and end_idx != -1:
        print(f"Replacing lines {start_idx+1} to {end_idx+1}")
        # We want to keep the section title which is before start_idx
        # The section title is at line 1198. start_idx should be around 1200.
        
        # Construct new ns.tex
        final_lines = ns_lines[:start_idx] + [new_content] + ns_lines[end_idx+1:]
        
        with open('ns.tex', 'w', encoding='utf-8') as f:
            f.writelines(final_lines)
        print("Successfully updated ns.tex")
    else:
        print("Could not find the block to replace in ns.tex")
        print(f"Start found: {start_idx}, End found: {end_idx}")

if __name__ == "__main__":
    process_tex()
