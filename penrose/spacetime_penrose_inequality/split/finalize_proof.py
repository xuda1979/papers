import os

def final_repair(path):
    with open(path, 'rb') as f:
        data = f.read()
    
    # Use byte strings to avoid ANY encoding/backslash ambiguity
    # These are the specific corruptions we saw in the logs
    replaces = [
        (b'\b\b', b'\'),
        (b'\t\t', b'\'),
        (b'\b\begin', b'\begin'),
        (b'\t\textbf', b'\textbf'),
        (b'\t\tr', b'\tr'),
        (b'\t\\', b'\'),
        (b'\r_', b'\tr_'), # Fix common 'r_' that was '	r_'
        (b'egin{', b'\begin{'),
        (b'extbf{', b'\textbf{'),
        (b'ef{', b'\ref{'),
        (b'ar{', b'\bar{'),
        (b'\\\', b'\'), # Triple to single
        (b'\\', b'\'),   # Double to single
    ]
    
    new_data = data
    for old, new in replaces:
        new_data = new_data.replace(old, new)
        
    # Special pass for the 	r macro which is often mangled
    new_data = new_data.replace(b'\ tr_', b' \tr_')
    
    with open(path, 'wb') as f:
        f.write(new_data)

files = [
    'sec_01_introduction.tex',
    'sec_02_the_penrose_conjecture.tex',
    'sec_34_logical_structure_and_gap_closure.tex'
]

for f in files:
    if os.path.exists(f):
        final_repair(f)
