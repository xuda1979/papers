import glob

bs = chr(92)
nl = chr(10)

for file in glob.glob("sec_*.tex"):
    with open(file, "r") as f:
        text = f.read()
    
    text = text.replace(bs + "	extbf{", bs + "textbf{")
    text = text.replace("	extbf{", bs + "textbf{")
    
    with open(file, "w") as f:
        f.write(text)
