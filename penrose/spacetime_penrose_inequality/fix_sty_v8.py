import os

cleveref_content = r"""\ProvidesPackage{cleveref}[2023/04/10 dummy cleveref]
\DeclareOption*{}\ProcessOptions
\RequirePackage{hyperref}

ewcommand{\cref}[1]{ef{#1}}

ewcommand{\Cref}[1]{ef{#1}}

ewcommand{\cpageref}[1]{\pageref{#1}}

ewcommand{\Cpageref}[1]{\pageref{#1}}
\endinput
"""

framed_content = r"""\ProvidesPackage{framed}[2023/04/10 dummy framed]

ewenvironment{framed}{}{}

ewenvironment{oframed}{}{}
\endinput
"""

for path in ['.', 'split']:
    with open(os.path.join(path, 'cleveref.sty'), 'w') as f:
        f.write(cleveref_content)
    with open(os.path.join(path, 'framed.sty'), 'w') as f:
        f.write(framed_content)
