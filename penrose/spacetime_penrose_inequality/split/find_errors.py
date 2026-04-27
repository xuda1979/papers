import re
files = ["sec_01_introduction.tex", "sec_34_logical_structure_and_gap_closure.tex"]
for fname in files:
    with open(fname, "r") as f:
        for i, line in enumerate(f, 1):
            if "\b\begin" in line: print(f"{fname}:{i}: \b\begin")
            if "\b\bar" in line: print(f"{fname}:{i}: \b\bar")
            if "\t\textbf" in line: print(f"{fname}:{i}: \t\textbf")
            if "\t\tr" in line: print(f"{fname}:{i}: \t\tr")
            if "egin{" in line and "\begin{" not in line: print(f"{fname}:{i}: BROKEN BEGIN")
            if "extbf{" in line and "\textbf{" not in line: print(f"{fname}:{i}: BROKEN TEXTBF")
            if "ef{" in line and "\ref{" not in line: print(f"{fname}:{i}: BROKEN REF")
            if "tr_" in line and "\tr_" not in line: print(f"{fname}:{i}: BROKEN TR")
            if "ar{" in line and "\bar{" not in line: print(f"{fname}:{i}: BROKEN BAR")
