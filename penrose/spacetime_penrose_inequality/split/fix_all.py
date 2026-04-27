import glob

def clean_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    
    # We need to strip the raw control characters \x08 (backspace) and \x09 (tab)
    # The  was becoming backspace, 	 becoming tab, etc.
    # So if there are lingering b'x08', b'x09', b'\x0d' (carriage return), let's fix them.
    # Actually wait, let's look for b'' literally and b'	'.
    
    original = data
    data = data.replace(b'\x08begin', b'begin')
    data = data.replace(b'\x09textbf', b'textbf')
    data = data.replace(b'\x08bar', b'bar')
    data = data.replace(b'ref', b'ref') # \x0dref -> ref
    data = data.replace(b'\eqr\ref', b'\eqref')
    
    # Let's do a more robust approach:
    replacements = {
        b'\b\begin': b'\begin',
        b'\t\textbf': b'\textbf',
        b'\r\ref': b'\ref',
        b'\eqr\ref': b'\eqref',
        b'\b\bar': b'\bar',
        b'\t ': b' ', # lingering 	?
        b'\b ': b' ', 
        b'\reference': b'reference',
        b'\sumption': b'sumption',
        b'as\sume': b'assume',
        b'as\sumed': b'assumed',
        b'\reflects': b'reflects',
        b'\summary': b'summary',
        b'\eqr': b'\eqr', # just in case
    }
    
    for old, new in replacements.items():
        data = data.replace(old, new)
        
    # More fixes for single slash issues on words that shouldn't have them
    for word in [b'reference', b'sumption', b'reflects', b'summary', b'egin', b'extbf', b'ef{', b'ar{']:
        data = data.replace(b'\'+word, word)
        
    data = data.replace(b'\begin', b'\begin') # no op
    data = data.replace(b'b\begin', b'\begin')
    data = data.replace(b't\textbf', b'\textbf')
    data = data.replace(b'r\ref', b'\ref')
    data = data.replace(b'b\bar', b'\bar')

    if data != original:
        with open(path, 'wb') as f:
            f.write(data)
        print(f"Fixed {path}")

for f in glob.glob('*.tex'):
    clean_file(f)
