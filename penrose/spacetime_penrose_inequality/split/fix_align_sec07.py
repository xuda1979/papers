import os

with open("sec_07_the_p_harmonic_level_set_method.tex", "r") as f:
    text = f.read()

bs = chr(92)
newline = chr(10)

# Replace the entire block with something simpler and more robust
old_block = bs + "begin{align*}" + newline
old_block += "    " + bs + "tg_{ij} - " + bs + "delta_{ij} &= " + bs + "phi^4 " + bs + "bg_{ij} - " + bs + "delta_{ij} = (" + bs + "phi^4 - 1)" + bs + "delta_{ij} + " + bs + "phi^4(" + bs + "bg_{ij} - " + bs + "delta_{ij}) " + bs + bs + newline
old_block += "    &= " + bs + "frac{4A}{r} + O(r^{-2}) + (1 + O(r^{-1})) " + bs + "cdot O(r^{-" + bs + "tau}) " + bs + bs + newline
old_block += "    &= O(r^{-" + bs + "min(" + bs + "tau,1)}) = O(r^{-" + bs + "tau'})." + newline
old_block += bs + "end{align*}"

new_block = bs + "begin{equation*} " + bs + "begin{aligned} "
new_block += bs + "tg_{ij} - " + bs + "delta_{ij} &= " + bs + "phi^4 " + bs + "bg_{ij} - " + bs + "delta_{ij} = (" + bs + "phi^4 - 1)" + bs + "delta_{ij} + " + bs + "phi^4(" + bs + "bg_{ij} - " + bs + "delta_{ij}) " + bs + bs + newline
new_block += "& = " + bs + "frac{4A}{r} + O(r^{-2}) + (1 + O(r^{-1})) " + bs + "cdot O(r^{-" + bs + "tau}) " + bs + bs + newline
new_block += "& = O(r^{-" + bs + "min(" + bs + "tau,1)}) = O(r^{-" + bs + "tau'})." + newline
new_block += bs + "end{aligned} " + bs + "end{equation*}"

if old_block in text:
    text = text.replace(old_block, new_block)
else:
    # If indentation is different, use regex
    import re
    text = re.sub(r"\begin\{align\*\}[\s\S]*?\end\{align\*\}", new_block, text, count=1)

with open("sec_07_the_p_harmonic_level_set_method.tex", "w") as f:
    f.write(text)
