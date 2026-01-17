
import re
import os

def flatten_tex(main_file, output_file, root_dir='.'):
    """
    Reads main_file, replaces \input{...} and \include{...} with file contents,
    and writes to output_file. Ignores commented out inputs.
    """

    # improved regex to handle optional spaces, braces, and ensure it's not commented
    # This regex looks for \input{filename} or \include{filename}
    # It attempts to verify it's not preceded by %
    # Note: simple regex is imperfect for complex LateX with % inside commands, but good enough for typical inputs
    input_pattern = re.compile(r'^\s*\\(input|include)\{([^}]+)\}')
    
    # Track processed files to avoid infinite loops if circular includes exist (shouldn't happening)
    processed_files = set()

    def read_and_process(filepath, current_depth=0):
        if current_depth > 10:
            return f"% MAX DEPTH EXCEEDED: {filepath}\n"
        
        full_path = os.path.join(root_dir, filepath)
        
        # Handle cases where extension .tex might be missing
        if not os.path.exists(full_path) and os.path.exists(full_path + '.tex'):
            full_path += '.tex'
            
        if not os.path.exists(full_path):
            print(f"Warning: File not found: {full_path}")
            return f"% FILE NOT FOUND: {filepath}\n"
            
        print(f"Processing: {filepath}")
        
        content = []
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            return f"% ERROR READING FILE {filepath}: {str(e)}\n"

        for line in lines:
            # Check for input command
            # We strip whitespace for the regex match, but we need to identify comments
            # If the line starts with % (ignoring whitespace), it's a comment, ignore.
            if line.strip().startswith('%'):
                content.append(line)
                continue

            # Check if there is an input command on this line
            # We look for the pattern. If found, we check if it is commented out inline (harder)
            # For this specific project, inputs are usually on their own lines.
            
            match = input_pattern.search(line)
            if match:
                # Check if there is a % before the match in the line
                start_index = match.start()
                preceding_text = line[:start_index]
                if '%' in preceding_text:
                    # It is commented out
                    content.append(line)
                else:
                    included_file = match.group(2)
                    # Recursively process
                    content.append(f"% Begin input: {included_file}\n")
                    content.append(read_and_process(included_file, current_depth + 1))
                    content.append(f"% End input: {included_file}\n")
            else:
                content.append(line)
        
        return "".join(content)

    final_content = read_and_process(main_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_content)
    
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    flatten_tex('main.tex', 'main_arxiv_submission.tex')
