import sys

# We use chr(92) for backslash to avoid any escaping issues in the python script generation itself.
bs = chr(92)

cleveref_content = [
    f"{bs}ProvidesPackage{{cleveref}}[2023/04/10 dummy cleveref]",
    f"{bs}DeclareOption*{{}}{bs}ProcessOptions",
    f"{bs}RequirePackage{{hyperref}}",
    f"{bs}newcommand{{{bs}cref}}[1]{{{bs}ref{{#1}}}}",
    f"{bs}newcommand{{{bs}Cref}}[1]{{{bs}ref{{#1}}}}",
    f"{bs}newcommand{{{bs}cpageref}}[1]{{{bs}pageref{{#1}}}}",
    f"{bs}newcommand{{{bs}Cpageref}}[1]{{{bs}pageref{{#1}}}}",
    f"{bs}endinput"
]

framed_content = [
    f"{bs}ProvidesPackage{{framed}}[2023/04/10 dummy framed]",
    f"{bs}newenvironment{{framed}}{{}}{{}}",
    f"{bs}newenvironment{{oframed}}{{}}{{}}",
    f"{bs}endinput"
]

with open("cleveref.sty", "w") as f:
    for line in cleveref_content:
        f.write(line + "
")

with open("framed.sty", "w") as f:
    for line in framed_content:
        f.write(line + "
")
