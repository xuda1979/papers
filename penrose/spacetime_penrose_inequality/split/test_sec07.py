with open("sec_07_the_p_harmonic_level_set_method.tex", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "min(" in line and "tau" in line:
        print(f"Line {i+1}: {repr(line)}")
