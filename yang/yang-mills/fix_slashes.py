with open("split/app174_holographic_stochastic_transport.tex", "r") as f:
    text = f.read()

# Replace actual control characters that got messed up in string literals
import string
clean_text = ""
for char in text:
    if char in string.printable and char not in "	
\x0b\x0c":
        clean_text += char
    elif char == '	':
        clean_text += '\t'
    elif char == '\x0c':
        clean_text += '\f'
    elif char == '\x08':
        clean_text += '\b'
    elif char == '
':
        clean_text += '
'
    else:
        clean_text += char

clean_text = clean_text.replace("\textbf", "\textbf")
clean_text = clean_text.replace(" egin", "\begin")
clean_text = clean_text.replace("\frac", "\frac")
clean_text = clean_text.replace("\tau", "\tau")
clean_text = clean_text.replace("\to", "\to")

with open("split/app174_holographic_stochastic_transport.tex", "w") as f:
    f.write(clean_text)
