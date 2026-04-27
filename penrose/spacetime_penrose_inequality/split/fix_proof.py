with open("proof_text.tex", "r") as f:
    text = f.read()

# Replace wrong characters from previous script
text = text.replace("ef{", "\ref{")
text = text.replace("egin{", "\begin{")
text = text.replace("heta^+", "\theta^+")
text = text.replace("r_\Sigma", "\tr_\Sigma")
text = text.replace("ext{", "\text{")
text = text.replace("abla", "\nabla")
text = text.replace("extbf{", "\textbf{")
text = text.replace("	", "")

with open("proof_text.tex", "w") as f:
    f.write(text)

