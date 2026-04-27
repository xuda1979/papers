import glob

def clean_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    
    original = data
    BS = bytes([92])
    
    replacements = {
        BS + b'noinden': BS + b'noindent',
        BS + b'barriers': b'barriers',
        BS + b'refined': b'refined',
        b'ca' + BS + b'reful': b'careful',
        b'The' + BS + b'refore': b'Therefore',
        BS + b'summarize': b'summarize',
        b'as' + BS + b'suming': b'assuming',
        b'As' + BS + b'sume': b'Assume',
    }
    
    for old, new in replacements.items():
        data = data.replace(old, new)
        
    if data != original:
        with open(path, 'wb') as f:
            f.write(data)
        print(f"Fixed {path}")

for f in glob.glob('*.tex'):
    clean_file(f)
