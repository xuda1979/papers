#!/usr/bin/env python3
"""
Script to split paper_clean.tex into multiple files by section.
Each section goes into its own file in the 'split' subdirectory.
A main.tex file is created that includes all the split files.
"""

import re
import os

def split_latex_paper(input_file, output_dir):
    """Split LaTeX paper into multiple files by section."""
    
    # Read the entire file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the document beginning and end
    doc_begin = content.find('\\begin{document}')
    doc_end = content.find('\\end{document}')
    
    if doc_begin == -1 or doc_end == -1:
        raise ValueError("Could not find \\begin{document} or \\end{document}")
    
    # Split content into preamble, body, and end
    preamble = content[:doc_begin + len('\\begin{document}')]
    body = content[doc_begin + len('\\begin{document}'):doc_end]
    ending = content[doc_end:]
    
    # Find all sections (both \section and \section*)
    section_pattern = r'\\section(\*?)\{([^}]+)\}(\s*\\label\{[^}]+\})?'
    sections = list(re.finditer(section_pattern, body))
    
    if not sections:
        raise ValueError("No sections found in document")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract front matter (abstract, maketitle, tableofcontents, etc.)
    first_section_pos = sections[0].start()
    front_matter = body[:first_section_pos].strip()
    
    # Split into individual sections
    section_files = []
    
    for i, match in enumerate(sections):
        # Determine the content of this section
        start = match.start()
        if i < len(sections) - 1:
            end = sections[i + 1].start()
        else:
            end = len(body)
        
        section_content = body[start:end].strip()
        
        # Create a filename from the section title
        section_title = match.group(2)
        is_starred = match.group(1) == '*'
        
        # Clean up the section title for filename
        # Remove LaTeX commands and special characters
        filename_base = re.sub(r'\\texorpdfstring\{([^}]+)\}\{[^}]*\}', r'\1', section_title)
        filename_base = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', filename_base)
        filename_base = re.sub(r'[\\${}^_]', '', filename_base)
        filename_base = re.sub(r'[^\w\s-]', '', filename_base)
        filename_base = re.sub(r'[-\s]+', '_', filename_base)
        filename_base = filename_base.strip('_').lower()
        
        # Limit filename length and add section number
        filename_base = filename_base[:60]
        filename = f"sec_{i+1:02d}_{filename_base}.tex"
        
        section_files.append(filename)
        
        # Write section to file
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(section_content + '\n')
        
        print(f"Created: {filename} ({len(section_content)} characters)")
    
    # Create the main.tex file
    main_file_path = os.path.join(output_dir, 'main.tex')
    with open(main_file_path, 'w', encoding='utf-8') as f:
        # Write preamble
        f.write(preamble + '\n\n')
        
        # Write front matter
        if front_matter:
            f.write(front_matter + '\n\n')
        
        # Include all section files
        for section_file in section_files:
            f.write(f'\\input{{{section_file[:-4]}}}  % {section_file}\n')
        
        # Write ending
        f.write('\n' + ending)
    
    print(f"\nCreated main file: {main_file_path}")
    print(f"Total sections split: {len(section_files)}")
    return main_file_path, section_files

if __name__ == '__main__':
    input_file = 'paper_clean.tex'
    output_dir = 'split'
    
    print(f"Splitting {input_file}...")
    print(f"Output directory: {output_dir}/")
    print("-" * 60)
    
    try:
        main_file, section_files = split_latex_paper(input_file, output_dir)
        print("-" * 60)
        print("✓ Successfully split paper!")
        print(f"✓ Compile with: cd split && pdflatex main.tex")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise
