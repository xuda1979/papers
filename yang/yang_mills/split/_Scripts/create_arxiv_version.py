"""
Create a single .tex file for arXiv submission by flattening all \input commands
"""

import os
import re
from pathlib import Path

def read_tex_file(filepath):
    """Read a .tex file with UTF-8 encoding"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        with open(filepath, 'r', encoding='latin-1') as f:
            return f.read()

def process_input_command(match, base_dir):
    """Process an \input{filename} command"""
    filename = match.group(1)
    
    # Add .tex extension if not present
    if not filename.endswith('.tex'):
        filename += '.tex'
    
    # Resolve the full path
    full_path = os.path.join(base_dir, filename)
    
    if not os.path.exists(full_path):
        print(f"Warning: File not found: {full_path}")
        return f"% FILE NOT FOUND: {filename}\n"
    
    print(f"Processing: {filename}")
    
    # Read the file content
    content = read_tex_file(full_path)
    
    # Add a comment marker
    result = f"\n% ========================================\n"
    result += f"% BEGIN: {filename}\n"
    result += f"% ========================================\n\n"
    result += content
    result += f"\n\n% ========================================\n"
    result += f"% END: {filename}\n"
    result += f"% ========================================\n\n"
    
    return result

def flatten_tex_file(input_file, output_file):
    """Flatten a .tex file by recursively expanding all \input commands"""
    
    base_dir = os.path.dirname(input_file)
    content = read_tex_file(input_file)
    
    # Process \input commands
    max_iterations = 20
    iteration = 0
    
    while r'\input{' in content and iteration < max_iterations:
        iteration += 1
        print(f"\nIteration {iteration}:")
        
        # Find all \input commands
        def replacer(match):
            return process_input_command(match, base_dir)
        
        # Replace \input{filename} with file contents
        content = re.sub(r'\\input\{([^}]+)\}', replacer, content)
    
    if iteration >= max_iterations:
        print("\nWarning: Maximum iterations reached. Some files may not be expanded.")
    
    # Remove \printbibliography commands (we'll add one at the end)
    content = re.sub(r'\\printbibliography\[.*?\]', '', content)
    content = re.sub(r'\\printbibliography', '', content)
    
    # Add bibliography at the end (before \end{document})
    content = re.sub(
        r'(\\end\{document\})',
        r'\\printbibliography\n\n\1',
        content
    )
    
    # Write the output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nFlattened file written to: {output_file}")
    print(f"Total length: {len(content)} characters")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    split_dir = script_dir.parent
    
    input_file = split_dir / "main.tex"
    output_file = split_dir / "main_arxiv.tex"
    
    print("=" * 60)
    print("Creating arXiv-ready single file version")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("=" * 60)
    
    flatten_tex_file(str(input_file), str(output_file))
    
    print("\nDone!")
