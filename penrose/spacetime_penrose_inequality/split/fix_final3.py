import re
files = ['sec_01_introduction.tex', 'sec_34_logical_structure_and_gap_closure.tex']
for fname in files:
    with open(fname, 'rb') as f:
        t = f.read().decode('utf-8')
    
    t = t.replace('\x08\begin', '\begin')
    t = t.replace('\x08\bar', '\bar')
    t = t.replace('	\textbf', '\textbf')
    t = t.replace('	\tr', '\tr')
    
    # Just literal string replacements to be safe
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
    
    # Remove duplicates
    t = t.replace('\\begin', '\begin')
    t = t.replace('\\textbf', '\textbf')
    t = t.replace('\\ref', '\ref')
    t = t.replace('\\tr', '\tr')
    t = t.replace('\\bar', '\bar')

    with open(fname, 'wb') as f:
        f.write(t.encode('utf-8'))
