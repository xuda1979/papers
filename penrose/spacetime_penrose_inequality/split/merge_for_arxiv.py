#!/usr/bin/env python3
"""
Merge all split .tex files into a single arxiv-ready file.
Replaces \input{...} commands with the actual file contents.
"""

import re
import os

def read_file(filepath):
    """Read file content with UTF-8 encoding."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def merge_tex_files(main_file, output_file):
    """Merge main.tex and all input files into a single file."""
    
    # Read the main file
    content = read_file(main_file)
    
    # Pattern to match \input{filename} commands
    input_pattern = re.compile(r'\\input\{([^}]+)\}')
    
    def replace_input(match):
        """Replace \input{file} with the file contents."""
        filename = match.group(1)
        if not filename.endswith('.tex'):
            filename += '.tex'
        
        filepath = os.path.join(os.path.dirname(main_file), filename)
        
        if os.path.exists(filepath):
            file_content = read_file(filepath)
            # Add a comment marker for debugging
            return f"\n% ========== BEGIN {filename} ==========\n{file_content}\n% ========== END {filename} ==========\n"
        else:
            print(f"Warning: File not found: {filepath}")
            return match.group(0)  # Keep original if file not found
    
    # Replace all \input commands
    merged_content = input_pattern.sub(replace_input, content)
    
    # Write the merged file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(merged_content)
    
    print(f"Created merged file: {output_file}")
    return merged_content

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_file = os.path.join(script_dir, 'main.tex')
    output_file = os.path.join(script_dir, 'arxiv_submission.tex')
    
    merge_tex_files(main_file, output_file)
    print("Done!")
