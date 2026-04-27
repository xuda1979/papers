import re

def clean_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # The actual strings we see in the Python script output
    text = text.replace('		extbf', r'	extbf')
    text = text.replace('		r', r'	r')
    
    # Also the ones identified as BROKEN by our check
    text = re.sub(r'\b\begin', r'\begin', text)
    text = re.sub(r'\b\bar', r'\bar', text)
    
    text = text.replace('egin{', r'egin{')
    text = text.replace('extbf{', r'	extbf{')
    text = text.replace('ef{', r'ef{')
    text = text.replace('ar{', r'ar{')

    # Replace specific `tr_` that are missing backslash
    text = re.sub(r'(?<!\)tr_', r'\tr_', text)

    # Double backslash cleanup
    text = text.replace(r'\begin', r'egin')
    text = text.replace(r'\textbf', r'	extbf')
    text = text.replace(r'\ref', r'ef')
    text = text.replace(r'\tr', r'	r')
    text = text.replace(r'\bar', r'ar')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

clean_file("sec_01_introduction.tex")
clean_file("sec_34_logical_structure_and_gap_closure.tex")
