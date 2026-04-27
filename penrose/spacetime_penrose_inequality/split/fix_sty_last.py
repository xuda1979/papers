import os

with open("cleveref.sty", "w") as f:
    f.write("\ProvidesPackage{cleveref}[2023/04/10 dummy cleveref]
")
    f.write("\DeclareOption*{}\ProcessOptions
")
    f.write("\RequirePackage{hyperref}
")
    f.write("\newcommand{\cref}[1]{\ref{#1}}
")
    f.write("\newcommand{\Cref}[1]{\ref{#1}}
")
    f.write("\newcommand{\cpageref}[1]{\pageref{#1}}
")
    f.write("\newcommand{\Cpageref}[1]{\pageref{#1}}
")
    f.write("\endinput
")

with open("framed.sty", "w") as f:
    f.write("\ProvidesPackage{framed}[2023/04/10 dummy framed]
")
    f.write("\newenvironment{framed}{}{}
")
    f.write("\newenvironment{oframed}{}{}
")
    f.write("\endinput
")

print("Files written successfully.")
