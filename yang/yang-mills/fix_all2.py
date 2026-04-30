import re

with open("split/app174_holographic_stochastic_transport.tex", "r") as f:
    text = f.read()

# using regex
text = re.sub(r'\x09extbf', r'\textbf', text)
text = re.sub(r'\x08egin', r'\begin', text)
text = re.sub(r'\x0cau', r'\tau', text)
text = re.sub(r'\x0co', r'\to', text)
text = re.sub(r'\x0crac', r'\frac', text)
text = re.sub(r'\x0dangle', r'\rangle', text)

with open("split/app174_holographic_stochastic_transport.tex", "w") as f:
    f.write(text)
