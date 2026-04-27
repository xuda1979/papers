import re
files = ['sec_01_introduction.tex', 'sec_34_logical_structure_and_gap_closure.tex']
for fname in files:
    with open(fname, 'rb') as f:
        t = f.read().decode('utf-8')
    # Use chr() to avoid backslash escaping issues in the script itself
    BS = chr(92)
    t = t.replace(chr(8), '')
    t = t.replace(chr(9), BS)
    t = t.replace('egin{', BS + 'begin{')
    t = t.replace('extbf{', BS + 'textbf{')
    t = t.replace('ef{', BS + 'ref{')
    t = t.replace('ar{', BS + 'bar{')
    t = t.replace('tr_', BS + 'tr_')
    t = t.replace(BS + BS + 'begin', BS + 'begin')
    t = t.replace(BS + BS + 'textbf', BS + 'textbf')
    t = t.replace(BS + BS + 'ref', BS + 'ref')
    t = t.replace(BS + BS + 'tr', BS + 'tr')
    t = t.replace(BS + BS + 'bar', BS + 'bar')
    with open(fname, 'wb') as f:
        f.write(t.encode('utf-8'))
