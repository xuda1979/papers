with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

text = text.replace('Theorem~
ef{thm:GapClosed}', r'Theorem~ef{thm:GapClosed}')
text = text.replace('Theorem~

ef{thm:GapClosed}', r'Theorem~ef{thm:GapClosed}')

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.write(text)
