import os

with open('framed.sty', 'w') as f:
    f.write(r'''\ProvidesPackage{framed}[2023/04/10 dummy framed]

ewenvironment{framed}{}{}

ewenvironment{oframed}{}{}
\endinput
''')

with open('cleveref.sty', 'w') as f:
    f.write(r'''\ProvidesPackage{cleveref}[2023/04/10 dummy cleveref]
\DeclareOption*{}\ProcessOptions
\RequirePackage{hyperref}

ewcommand{\cref}[1]{ef{#1}}

ewcommand{\Cref}[1]{ef{#1}}

ewcommand{\cpageref}[1]{\pageref{#1}}

ewcommand{\Cpageref}[1]{\pageref{#1}}
\endinput
''')
