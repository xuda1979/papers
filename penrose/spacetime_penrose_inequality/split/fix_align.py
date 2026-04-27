with open("sec_07_the_p_harmonic_level_set_method.tex", "r") as f:
    text = f.read()

# Let's look for the start of the align environment containing line 233
lines = text.splitlines()
start = max(0, 233 - 10)
end = min(len(lines), 233 + 5)
for i in range(start, end):
    print(f"{i+1}: {repr(lines[i])}")
