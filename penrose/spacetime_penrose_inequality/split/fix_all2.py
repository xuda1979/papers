import glob

def clean_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    
    original = data
    BS = bytes([92])
    
    replacements = {
        BS + b'b' + BS + b'begin': BS + b'begin',
        BS + b't' + BS + b'textbf': BS + b'textbf',
        BS + b'r' + BS + b'ref': BS + b'ref',
        BS + b'eqr' + BS + b'ref': BS + b'eqref',
        BS + b'b' + BS + b'bar': BS + b'bar',
        BS + b'reference': b'reference',
        BS + b'sumption': b'sumption',
        b'as' + BS + b'sume': b'assume',
        b'as' + BS + b'sumed': b'assumed',
        BS + b'reflects': b'reflects',
        BS + b'summary': b'summary'
    }
    
    for old, new in replacements.items():
        data = data.replace(old, new)
        
    for char in [b'b', b't', b'r']:
        data = data.replace(char + BS + b'begin', BS + b'begin')
        data = data.replace(char + BS + b'textbf', BS + b'textbf')
        data = data.replace(char + BS + b'ref', BS + b'ref')
        data = data.replace(char + BS + b'bar', BS + b'bar')

    if data != original:
        with open(path, 'wb') as f:
            f.write(data)
        print(f"Fixed {path}")

for f in glob.glob('*.tex'):
    clean_file(f)
