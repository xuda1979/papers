import os
import re

def flatten_tex(main_file_path, output_file_path):
    base_dir = os.path.dirname(main_file_path)
    
    with open(main_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find \input{...} or \include{...}
    # Handling potential spaces and different path formats
    include_pattern = re.compile(r'\\(?:input|include)\{([^}]+)\}')
    
    def replace_include(match):
        full_match = match.group(0)
        start_index = match.start()
        
        # Check if the line is commented out
        # Find the start of the line
        line_start = content.rfind('\n', 0, start_index) + 1
        line_content = content[line_start:start_index]
        if '%' in line_content:
            return full_match # Return as is, don't expand

        rel_path = match.group(1)
        # Handle cases where .tex extension might be missing
        if not rel_path.endswith('.tex'):
            rel_path += '.tex'
            
        full_path = os.path.join(base_dir, rel_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: File not found: {full_path}")
            return f"% File not found: {rel_path}\n"
        
        print(f"Including: {rel_path}")
        with open(full_path, 'r', encoding='utf-8') as f:
            included_content = f.read()
            
        # Recursive processing for nested inputs
        # We need to process included_content recursively, but using a fresh call 
        # Here we just do one level for simplicity as per previous logic, 
        # but to be robust we should recurse. 
        # Let's keep it simple: just expand. If nested inputs exist they won't be expanded 
        # with this simple regex sub unless we loop. 
        
        return f"\n% --- Start included file: {rel_path} ---\n{included_content}\n% --- End included file: {rel_path} ---\n"

    flattened_content = include_pattern.sub(replace_include, content)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(flattened_content)
    
    print(f"Flattened LaTeX file created at: {output_file_path}")

if __name__ == "__main__":
    try:
        # relative paths assuming running from yang_mills/split
        main_tex = "main.tex"
        output_tex = "arxiv_submission.tex"
        
        print(f"Working directory: {os.getcwd()}")
        print(f"Target main file: {os.path.abspath(main_tex)}")
        
        if not os.path.exists(main_tex):
            print("Error: main.tex not found in current directory.")
            exit(1)
            
        flatten_tex(main_tex, output_tex)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
