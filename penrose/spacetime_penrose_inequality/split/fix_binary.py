import re

def fix_file(filename):
    with open(filename, "rb") as f:
        data = f.read()

    data = data.replace(b'\x08egin', b'\begin')
    data = data.replace(b'\x08ar', b'\bar')
    data = data.replace(b'\x09extbf', b'\textbf')
    data = data.replace(b'\x09r', b'\tr')
    data = data.replace(b'\x0Def', b'\ref')
    data = data.replace(b'\x0Doindent', b'\noindent')
    
    text = data.decode('latin1')
    
    # Simple direct string replacements for the broken pieces
    # Use double-backslash in Python strings to represent a single backslash for the file
    text = text.replace('
ef{', '
\ref{')
    text = text.replace('
ef{', '
\ref{')
    text = text.replace('ef{', '\ref{') # If ref{ was already broken but on same line
    
    text = text.replace('egin{', '\begin{')
    text = text.replace('oindent', '\noindent')
    text = text.replace('extbf{', '\textbf{')
    text = text.replace('r_', '\tr_')
    text = text.replace('ar{', '\bar{')
    
    # Cleanup double backslashes
    text = text.replace('\\begin', '\begin')
    text = text.replace('\\ref', '\ref')
    text = text.replace('\\noindent', '\noindent')
    text = text.replace('\\tr', '\tr')
    text = text.replace('\\bar', '\bar')
    text = text.replace('\\textbf', '\textbf')
    text = text.replace('\\end', '\end')

    with open(filename, "wb") as f:
        f.write(text.encode('latin1'))

fix_file("sec_01_introduction.tex")
fix_file("sec_34_logical_structure_and_gap_closure.tex")
