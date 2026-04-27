with open("sec_03_overview.tex", "r") as f:
    text = f.read()

text = text.replace("remains an resolved problem", "remains an open problem")

with open("sec_03_overview.tex", "w") as f:
    f.write(text)
