with open("main.tex", "r") as f:
    text = f.read()

bs = chr(92)
nl = chr(10)
text = text.replace(bs + "	extbf{", bs + "textbf{")

lines = text.splitlines()
for i, line in enumerate(lines):
    if "extbf{Theorem C" in line:
        lines[i] = lines[i].replace("	extbf{", bs + "textbf{")
    if "neq 0" in line and bs + "neq 0" not in line:
        lines[i] = lines[i].replace("neq 0", bs + "neq 0")

with open("main.tex", "w") as f:
    f.write(nl.join(lines))
