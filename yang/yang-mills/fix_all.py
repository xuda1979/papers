with open("split/app174_holographic_stochastic_transport.tex", "r") as f:
    text = f.read()

# Manual string replacements to avoid python parser issues
text = text.replace('	', '\t') # fix tabs
text = text.replace('\x08', '\b') # fix backspaces
text = text.replace('\x0c', '\f') # fix form feeds
text = text.replace('', '\r') # fix carriage returns

text = text.replace('\textbf', '\textbf')
text = text.replace(' egin', '\begin')
text = text.replace('\tau', '\tau')
text = text.replace('\to', '\to')
text = text.replace('\frac', '\frac')

with open("split/app174_holographic_stochastic_transport.tex", "w") as f:
    f.write(text)
