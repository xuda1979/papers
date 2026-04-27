import os

def fix_file(path, replacements):
    if not os.path.exists(path):
        return
    with open(path, 'r') as f:
        content = f.read()
    
    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)
    
    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"Updated {path}")

# Specifically fix the env tags in sec_15_conclusion_and_outlook.tex
fix_file('sec_15_conclusion_and_outlook.tex', [
    (r'egin{openproblem}[Variational-to-Pointwise Favorable Jump Gap]', r'egin{proposition}[Variational-to-Pointwise Favorable Jump Resolution]'),
    (r'egin{openproblem}[Variational-to-Pointwise Favorable Jump Resolution]', r'egin{proposition}[Variational-to-Pointwise Favorable Jump Resolution]'),
    (r'\end{openproblem}', r'\end{proposition}')
])
