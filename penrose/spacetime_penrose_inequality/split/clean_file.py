import re

def clean_file(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()
    
    # 1. Remove binary control characters like backspace \x08
    content = content.replace(b'\x08', b'')
    
    # 2. Decode to text
    text = content.decode('utf-8')
    
    # 3. Fix broken egin blocks which might have become 'egin{' or '\ egin{'
    text = re.sub(r'\? ?egin\{', r'\begin{', text)
    
    # 4. Fix broken 
abla which might have become '
 
abla' or ' 
abla' or '

abla'
    text = text.replace('

abla', '\nabla')
    text = text.replace('
 
abla', '\nabla')
    
    with open(filepath, 'w') as f:
        f.write(text)

clean_file('sec_34_logical_structure_and_gap_closure.tex')
clean_file('sec_36_dispersive_estimates_and_spectral_transfer.tex')
