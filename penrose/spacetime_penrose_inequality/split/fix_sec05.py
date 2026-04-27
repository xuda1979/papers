with open("sec_05_ricci_flow_inspired_monotonicity_formulas.tex", "r") as f:
    text = f.read()

text = text.replace("d{equation}", "\end{equation}")
text = text.replace("\	extbf{not generally monotone}", "\textbf{not generally monotone}")

with open("sec_05_ricci_flow_inspired_monotonicity_formulas.tex", "w") as f:
    f.write(text)
