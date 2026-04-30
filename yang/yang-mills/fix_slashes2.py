import os

file_path = "split/app174_holographic_stochastic_transport.tex"

with open(file_path, "r") as f:
    text = f.read()

# Fix common python escaping messups
text = text.replace("	extbf", "\textbf")
text = text.replace(" egin", "\begin")
text = text.replace("rac", "\frac")
text = text.replace("	au", "\tau")
text = text.replace(" 	o ", " \to ")
text = text.replace("	heta", "\theta")
text = text.replace(" ef", "\ref")
text = text.replace("angle", "\rangle")
text = text.replace("	ilde", "\tilde")

with open(file_path, "w") as f:
    f.write(text)

