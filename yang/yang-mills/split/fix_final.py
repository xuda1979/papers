with open("/Users/daxu/papers/yang/yang-mills/split/app174_holographic_stochastic_transport.tex", "r") as f:
    text = f.read()

text = text.replace('BACKSLASH', chr(92))

with open("/Users/daxu/papers/yang/yang-mills/split/app174_holographic_stochastic_transport.tex", "w") as f:
    f.write(text)
