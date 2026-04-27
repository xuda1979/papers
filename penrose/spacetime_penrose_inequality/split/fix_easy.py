import re

def fix(fname):
    with open(fname, "rb") as f:
        data = f.read()

    # Just remove any backspaces 0x08 entirely
    data = data.replace(b'\x08', b'')
    
    text = data.decode('utf-8', errors='ignore')

    # Convert tabs back to 	 but only inside our macros if possible, 
    # actually it is easier to replace them unconditionally if they precede "extbf"
    text = re.sub(r'[	]+extbf', r'\textbf', text)
    text = re.sub(r'[	]+r', r'\tr', text)
    
    # Fix egin, oindent, ar, ef
    text = re.sub(r'\b\s*egin', r'\begin', text)
    text = re.sub(r'\b\s*ar', r'\bar', text)
    
    # In some places it's literally the letter b, then spaces, then egin.
    # Wait, the log says "^^H egin". Which means it's  followed by 0x08.
    # We stripped 0x08. So it's just  egin
    text = re.sub(r'\b\s*egin', r'\begin', text)
    text = re.sub(r'\b\s*ar', r'\bar', text)
    
    # Same for 	 extbf
    text = re.sub(r'\t\s*extbf', r'\textbf', text)
    text = re.sub(r'\t\s*r', r'\tr', text)
    
    # Same for 
 oindent
    # The log didn't complain about oindent with ^^H, but let's be safe.
    text = re.sub(r'\n\s*oindent', r'\noindent', text)
    
    # Same for  ef
    text = re.sub(r'\r\s*ef', r'\ref', text)

    # Some basic literal replacements for common errors from log
    text = text.replace('egin{', 'egin{')
    text = text.replace('extbf{', '	extbf{')
    text = text.replace('oindent', '
oindent')
    text = text.replace('ef{', 'ef{')
    text = text.replace('r_', '	r_')
    text = text.replace('ar{', 'ar{')
    
    # Cleanup double escapes
    text = text.replace('\begin', 'egin')
    text = text.replace('\textbf', '	extbf')
    text = text.replace('\noindent', '
oindent')
    text = text.replace('\ref', 'ef')
    text = text.replace('\tr', '	r')
    text = text.replace('\bar', 'ar')

    with open(fname, "w", encoding='utf-8') as f:
        f.write(text)

fix("sec_01_introduction.tex")
fix("sec_34_logical_structure_and_gap_closure.tex")
