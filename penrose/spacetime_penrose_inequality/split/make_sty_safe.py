import sys

# We need literal backslashes in the output.
cleveref_lines = [
    r"\ProvidesPackage{cleveref}[2023/04/10 dummy cleveref]",
    r"\DeclareOption*{}\ProcessOptions",
    r"\RequirePackage{hyperref}",
    r"
ewcommand{\cref}[1]{ef{#1}}",
    r"
ewcommand{\Cref}[1]{ef{#1}}",
    r"
ewcommand{\cpageref}[1]{\pageref{#1}}",
    r"
ewcommand{\Cpageref}[1]{\pageref{#1}}",
    r"\endinput"
]

framed_lines = [
    r"\ProvidesPackage{framed}[2023/04/10 dummy framed]",
    r"
ewenvironment{framed}{}{}",
    r"
ewenvironment{oframed}{}{}",
    r"\endinput"
]

with open("cleveref.sty", "w") as f:
    for line in cleveref_lines:
        f.write(line + "
")

with open("framed.sty", "w") as f:
    for line in framed_lines:
        f.write(line + "
")
