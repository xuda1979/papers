import glob

bs = chr(92)
nl = chr(10)

for file in glob.glob("sec_*.tex"):
    with open(file, "r") as f:
        text = f.read()
    
    text = text.replace(" ef{", " " + bs + "ref{")
    text = text.replace("~ef{", "~" + bs + "ref{")
    text = text.replace("(ef{", "(" + bs + "ref{")
    text = text.replace("{ef{", "{" + bs + "ref{")
    text = text.replace(nl + "ef{", nl + bs + "ref{")
    text = text.replace("k eq 0", "k " + bs + "neq 0")
    text = text.replace("k " + nl + "eq 0", "k " + bs + "neq 0")

    with open(file, "w") as f:
        f.write(text)

with open("main.tex", "r") as f:
    text = f.read()

text = text.replace(" ef{", " " + bs + "ref{")
text = text.replace("~ef{", "~" + bs + "ref{")
text = text.replace("(ef{", "(" + bs + "ref{")
text = text.replace("{ef{", "{" + bs + "ref{")
text = text.replace(nl + "ef{", nl + bs + "ref{")

with open("main.tex", "w") as f:
    f.write(text)
