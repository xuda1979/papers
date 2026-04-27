with open("sec_05_ricci_flow_inspired_monotonicity_formulas.tex", "r") as f:
    text = f.read()

text = text.replace("d{remark}", "\end{remark}")
text = text.replace("d{itemize}", "\end{itemize}")

with open("sec_05_ricci_flow_inspired_monotonicity_formulas.tex", "w") as f:
    f.write(text)
