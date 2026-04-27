import glob
import os

def clean_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    
    original = data
    BS = bytes([92])
    
    data = data.replace(BS + b'noindentt', BS + b'noindent')
    data = data.replace(b'as' + BS + b'suming', b'assuming')
    data = data.replace(b'As' + BS + b'sume', b'Assume')
    data = data.replace(b'as' + BS + b'sume', b'assume')
    data = data.replace(b'As' + BS + b'suming', b'Assuming')
    data = data.replace(BS + b'suming', b'suming')
    data = data.replace(BS + b'sume', b'sume')
    data = data.replace(BS + b'sumption', b'sumption')
    
    if data != original:
        with open(path, 'wb') as f:
            f.write(data)
        print(f"Fixed {path}")

for f in glob.glob('*.tex'):
    clean_file(f)
