with open("proof_text.tex", "rb") as f:
    text = f.read()

# Replace actual control characters
text = text.replace(b'', b'\r')
text = text.replace(b'
', b'\n')
text = text.replace(b'	', b'\t')
text = text.replace(b'\x08', b'\b')

with open("proof_text2.tex", "wb") as f:
    f.write(text)

