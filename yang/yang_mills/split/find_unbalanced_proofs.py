import os
import re

def check_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove comments
    content = re.sub(r'%.*', '', content)
    
    begins = content.count(r'\begin{proof}')
    ends = content.count(r'\end{proof}')
    
    if begins != ends:
        print(f"File: {filepath} has {begins} begins and {ends} ends.")
        return False
    return True

root_dir = r"c:\Users\Lenovo\papers\yang\yang_mills\split"
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".tex"):
            check_file(os.path.join(dirpath, filename))
