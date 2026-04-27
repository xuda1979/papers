import os

with open('sec_01_introduction.tex', 'rb') as f:
    text = f.read()

# Let's fix those last missing backslashes on commands
# begin -> egin
text = text.replace(b'\b\begin', b'\begin')
text = text.replace(b'\t\textbf', b'\textbf')
text = text.replace(b'\r\ref', b'\ref')
text = text.replace(b'\eqr\ref', b'\eqref')
text = text.replace(b'\b\bar', b'\bar')

with open('sec_01_introduction.tex', 'wb') as f:
    f.write(text)
