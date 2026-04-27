import glob
import re

for file in glob.glob("sec_*.tex"):
    with open(file, "r") as f:
        text = f.read()
    
    # Are there any lingering "ef{"?
    if "ef{" in text and "\ref{" not in text and "\eqref{" not in text:
        print(f"File {file} has lingering ef{{")
        
    # Are there any "eq 0" which should be "neq 0" or "= 0"?
    # Just print the context to check manually.
    if "eq 0" in text and "neq" not in text:
        print(f"File {file} might have a broken 'eq 0'")
