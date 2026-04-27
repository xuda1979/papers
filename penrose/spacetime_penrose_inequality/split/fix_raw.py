import re

def fix(fname):
    with open(fname, "rb") as f:
        data = f.read()

    # Just remove any backspaces 0x08 entirely
    data = data.replace(b'\x08', b'')
    
    text = data.decode('utf-8', errors='ignore')

    # Fix egin, oindent, ar, ef
    text = text.replace('egin{', '\begin{')
    text = text.replace('extbf{', '\textbf{')
    text = text.replace('oindent', '\noindent')
    text = text.replace('ef{', '\ref{')
    text = text.replace('r_', '\tr_')
    text = text.replace('ar{', '\bar{')
    
    # Cleanup double escapes
    text = text.replace('\\begin', '\begin')
    text = text.replace('\\textbf', '\textbf')
    text = text.replace('\\noindent', '\noindent')
    text = text.replace('\\ref', '\ref')
    text = text.replace('\\tr', '\tr')
    text = text.replace('\\bar', '\bar')

    with open(fname, "w", encoding='utf-8') as f:
        f.write(text)

fix("sec_01_introduction.tex")
fix("sec_34_logical_structure_and_gap_closure.tex")
