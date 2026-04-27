with open("sec_34_logical_structure_and_gap_closure.tex", "rb") as f:
    data = f.read()
# Replace backspace+egin with egin
data = data.replace(b"\x08egin", b"\begin")
with open("sec_34_logical_structure_and_gap_closure.tex", "wb") as f:
    f.write(data)
