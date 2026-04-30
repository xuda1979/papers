with open("split/app174_holographic_stochastic_transport.tex", "r") as f:
    text = f.read()

# Replace all literal escape characters with actual slashes
text = text.replace(r"\", r"\ ")
text = text.replace(r"\ ", r"\ ")
text = text.replace(r"\", "\")
text = text.replace(r"
", "
")

text = text.replace(r"\x08egin", r"egin")
text = text.replace(r"	o", r"	o")
text = text.replace(r"\x0crac", r"rac")
text = text.replace("
angle", r"angle")
text = text.replace(r"\", "\")

with open("split/app174_holographic_stochastic_transport.tex", "w") as f:
    f.write(text)
