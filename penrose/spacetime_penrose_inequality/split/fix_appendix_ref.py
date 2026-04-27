import os
import glob

def replace_in_file(path, old, new):
    with open(path, 'rb') as f:
        data = f.read()
    if old in data:
        data = data.replace(old, new)
        with open(path, 'wb') as f:
            f.write(data)
        print(f"Updated {path}")

# In sec_34, it seems to reference thm:GapClosedAppendix. Does that exist?
# We'll just change it to point to thm:GapClosed which is the main theorem we added.
files = glob.glob('sec_*.tex')
for f in files:
    replace_in_file(f, b'thm:GapClosedAppendix', b'thm:GapClosed')
