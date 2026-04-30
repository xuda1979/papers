import re

with open('/Users/daxu/papers/yang/yang-mills/split/app174_holographic_stochastic_transport.tex', 'r') as f:
    text = f.read()

text = text.replace('\x08', '\b')
text = text.replace('\x0c', '\f')
text = text.replace('	', '\t')
text = text.replace('', '\r')

with open('/Users/daxu/papers/yang/yang-mills/split/app174_holographic_stochastic_transport.tex', 'w') as f:
    f.write(text)
