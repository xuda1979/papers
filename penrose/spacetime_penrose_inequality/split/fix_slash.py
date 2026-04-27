import os
import re

def fix(path):
    with open(path, 'r', encoding='utf-8') as f:
        t = f.read()

    # The logs indicate we have raw  and 	 control characters inside the file
    t = t.replace('\x08', '') # Backspace
    t = t.replace('\x09\x09', '\') # Double tab
    t = t.replace('\x09\', '\')   # Tab + slash
    t = t.replace('\x09', '')       # Single tab

    # Fix the missing backslashes that caused the words to become "egin", "extbf"
    t = t.replace('egin{', '\begin{')
    t = t.replace('extbf{', '\textbf{')
    t = t.replace('ef{', '\ref{')
    t = t.replace('ar{', '\bar{')
    t = t.replace(' tr_', ' \tr_')
    t = t.replace('
tr_', '
\tr_')
    t = t.replace('{tr_', '{\tr_')
    t = t.replace('(tr_', '(\tr_')
    t = t.replace('-tr_', '-\tr_')
    t = t.replace('=tr_', '=\tr_')
    
    # Clean up double slashes from the replace
    t = t.replace('\\begin', '\begin')
    t = t.replace('\\textbf', '\textbf')
    t = t.replace('\\ref', '\ref')
    t = t.replace('\\tr', '\tr')
    t = t.replace('\\bar', '\bar')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(t)

for f in ['sec_01_introduction.tex', 'sec_34_logical_structure_and_gap_closure.tex']:
    if os.path.exists(f):
        fix(f)
