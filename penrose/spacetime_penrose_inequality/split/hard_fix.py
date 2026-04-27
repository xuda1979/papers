import re

def fix(fname):
    with open(fname, "rb") as f:
        data = f.read()

    # The raw broken characters:
    # 0x08 = backspace, 0x09 = tab, 0x0A = newline, 0x0D = carriage return
    # Some combinations got stuck in the file

    # 1. Clean backspaces explicitly if any left
    # But ONLY in our corrupted macro words to be safe
    data = data.replace(b'b\x08egin', b'\begin')
    data = data.replace(b'\x08egin', b'\begin')
    data = data.replace(b'b\x08ar', b'\bar')
    data = data.replace(b'\x08ar', b'\bar')
    
    # 2. tab followed by "extbf" or "r"
    data = data.replace(b't\x09extbf', b'\textbf')
    data = data.replace(b'\x09extbf', b'\textbf')
    data = data.replace(b't\x09r', b'\tr')
    data = data.replace(b'\x09r', b'\tr')
    
    text = data.decode('utf-8', errors='ignore')

    # Look for broken refs
    text = re.sub(r'~[
]*ef\{', r'~\ref{', text)
    text = re.sub(r' ([
]+)ef\{', r' \1\ref{', text)

    # Broken macros
    text = re.sub(r'[
]+egin\{', r'\begin{', text)
    text = re.sub(r' ([
]+)egin\{', r' \1\begin{', text)
    text = re.sub(r'[
]+oindent', r'\noindent', text)
    text = re.sub(r' ([
]+)oindent', r' \1\noindent', text)

    text = re.sub(r'\begin', r'\begin', text)

    # Some remaining seen in log
    text = text.replace('\b^^H
         egin', '\begin')
    text = text.replace('\b^^H
          egin', '\begin')
    text = text.replace('\b^^H
           egin', '\begin')
    text = text.replace('\b^^H
        egin', '\begin')
    
    # Re-apply manual string replaces
    text = text.replace('^^H
        egin', '\begin')
    text = text.replace('^^H
         egin', '\begin')
    text = text.replace('^^H
          egin', '\begin')
    text = text.replace('^^H
           egin', '\begin')

    # Also those with  prefix
    text = text.replace('\b\x08egin', '\begin')
    text = text.replace('\b\x08ar', '\bar')

    # Replace literally:
    text = text.replace('
egin', '\begin')
    text = text.replace('
 ar', '\bar')
    
    # The actual string representations from the logs:
    text = text.replace('\b
         egin', '\begin')
    text = text.replace('\b
          egin', '\begin')
    text = text.replace('\b
           egin', '\begin')
    text = text.replace('\b
        egin', '\begin')

    text = text.replace('^^H
        ar', '\bar')
    text = text.replace('^^H
         ar', '\bar')
    text = text.replace('\b^^H
                  ar', '\bar')
    text = text.replace('\b^^H
                    ar', '\bar')
    text = text.replace('\b^^H
                             ar', '\bar')

    # Find ANY `^^H` or `^^H` and clean it
    text = re.sub(r'\b\x08\s*egin', r'\begin', text)
    text = re.sub(r'\x08\s*egin', r'\begin', text)
    text = re.sub(r'\b\x08\s*ar', r'\bar', text)
    text = re.sub(r'\x08\s*ar', r'\bar', text)
    
    text = re.sub(r'\t\x09\s*extbf', r'\textbf', text)
    text = re.sub(r'\x09\s*extbf', r'\textbf', text)
    
    text = re.sub(r'\t\x09\s*r', r'\tr', text)
    text = re.sub(r'\x09\s*r', r'\tr', text)

    # Some manual stuff from the logs
    text = text.replace('\t
	extbf', '\textbf')
    text = text.replace('\t
	r', '\tr')

    with open(fname, "w", encoding='utf-8') as f:
        f.write(text)

fix("sec_01_introduction.tex")
fix("sec_34_logical_structure_and_gap_closure.tex")
