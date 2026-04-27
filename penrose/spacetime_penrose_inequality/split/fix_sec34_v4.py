import os

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

bs = chr(92)
newline = chr(10)

lines = text.splitlines()
for i, line in enumerate(lines):
    # Fix the missing backslashes in tr_Sigma, int_Sigma, psi_1, and ge 0
    # but ONLY if they are not already preceded by a backslash
    # We use a simple replacement for these specific tokens in this file
    if "tr_Sigma" in line:
        line = line.replace("tr_Sigma", bs + "tr_Sigma")
    if "int_Sigma" in line:
        line = line.replace("int_Sigma", bs + "int_Sigma")
    if "psi_1" in line:
        line = line.replace("psi_1", bs + "psi_1")
    if "ge 0" in line:
        line = line.replace("ge 0", bs + "ge 0")
    
    # Clean up double backslashes if any were created
    line = line.replace(bs + bs + "tr_Sigma", bs + "tr_Sigma")
    line = line.replace(bs + bs + "int_Sigma", bs + "int_Sigma")
    line = line.replace(bs + bs + "psi_1", bs + "psi_1")
    line = line.replace(bs + bs + "ge 0", bs + "ge 0")
    
    lines[i] = line

text = newline.join(lines)

# Now fix the theorem body if it was already inserted with errors
if "label{thm:GapClosed}" in text:
    # Ensure the math parts have $ $
    text = text.replace(bs + "int_" + bs + "Sigma (" + bs + "tr_" + bs + "Sigma k) " + bs + "psi_1 " + bs + ", dA " + bs + "ge 0", 
                        "$" + bs + "int_" + bs + "Sigma (" + bs + "tr_" + bs + "Sigma k) " + bs + "psi_1 " + bs + ", dA " + bs + "ge 0$")
    text = text.replace(bs + "tr_" + bs + "Sigma k " + bs + "ge 0", 
                        "$" + bs + "tr_" + bs + "Sigma k " + bs + "ge 0$")
    # Clean up double $$ if any
    text = text.replace("$$", "$")

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.write(text)
