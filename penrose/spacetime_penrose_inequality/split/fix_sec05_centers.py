with open("sec_05_ricci_flow_inspired_monotonicity_formulas.tex", "r") as f:
    text = f.read()

bs = chr(92)
# Fix all d{...} and missing backslashes in sec_05
text = text.replace("d{enumerate}", bs + "end{enumerate}")
text = text.replace("d{center}", bs + "end{center}")
text = text.replace("d{equation}", bs + "end{equation}")
text = text.replace("d{align}", bs + "end{align}")

# Clean up backslash issues in textbf
text = text.replace(bs + "	extbf{", bs + "textbf{")

with open("sec_05_ricci_flow_inspired_monotonicity_formulas.tex", "w") as f:
    f.write(text)
