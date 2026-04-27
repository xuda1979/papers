with open("sec_05_ricci_flow_inspired_monotonicity_formulas.tex", "r") as f:
    text = f.read()

bs = chr(92)
text = text.replace("d{align}", bs + "end{align}")
text = text.replace(bs + "	extbf{", bs + "textbf{")

with open("sec_05_ricci_flow_inspired_monotonicity_formulas.tex", "w") as f:
    f.write(text)
