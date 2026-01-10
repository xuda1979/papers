import os
import re

def check_mismatched_environments(directory):
    tex_files = [f for f in os.listdir(directory) if f.endswith('.tex')]
    
    environments = {}
    
    for filename in tex_files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    content = f.read()
            except:
                print(f"Could not read {filename}")
                continue

        # Simple check for definition
        begins = len(re.findall(r'\\begin\{definition\}', content))
        ends = len(re.findall(r'\\end\{definition\}', content))
        if begins != ends:
            print(f"File: {filename} -> begin{{definition}}: {begins}, end{{definition}}: {ends}")

        # Simple check for proof
        begins_p = len(re.findall(r'\\begin\{proof\}', content))
        ends_p = len(re.findall(r'\\end\{proof\}', content))
        if begins_p != ends_p:
             print(f"File: {filename} -> begin{{proof}}: {begins_p}, end{{proof}}: {ends_p}")
        
        # Simple check for equation
        begins_e = len(re.findall(r'\\begin\{equation\}', content))
        ends_e = len(re.findall(r'\\end\{equation\}', content))
        if begins_e != ends_e:
             print(f"File: {filename} -> begin{{equation}}: {begins_e}, end{{equation}}: {ends_e}")

check_mismatched_environments('.')
