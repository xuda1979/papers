with open("/Users/daxu/papers/yang/yang-mills/split/app174_holographic_stochastic_transport.tex", "rb") as f:
    text = f.read()

text = text.replace(b'\x09extbf', b'\textbf')
text = text.replace(b'\x0aewtheorem', b'
\newtheorem')
text = text.replace(b'begin{', b'\begin{')
text = text.replace(b'\x09au', b'\tau')
text = text.replace(b'\x0crac', b'\frac')
text = text.replace(b'\x09o', b'\to')
text = text.replace(b'\x09heta', b'\theta')
text = text.replace(b'\x0aef{', b'
\ref{')
text = text.replace(b'\x0aangle', b'
\rangle')
text = text.replace(b'\x09ilde', b'\tilde')
text = text.replace(b'\x08egin', b'\begin')
text = text.replace(b'\x08', b'\b')

with open("/Users/daxu/papers/yang/yang-mills/split/app174_holographic_stochastic_transport.tex", "wb") as f:
    f.write(text)
