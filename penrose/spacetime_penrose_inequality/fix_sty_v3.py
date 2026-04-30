import os

def write_sty(name, content):
    with open(name, 'wb') as f:
        f.write(content.encode('ascii'))

# cleveref.sty
cleveref = r'''\ProvidesPackage{cleveref}[2023/04/10 dummy cleveref]
\DeclareOption*{}\ProcessOptions
\RequirePackage{hyperref}

ewcommand{\cref}[1]{ef{#1}}

ewcommand{\Cref}[1]{ef{#1}}

ewcommand{\cpageref}[1]{\pageref{#1}}

ewcommand{\Cpageref}[1]{\pageref{#1}}
\endinput
'''

# framed.sty
framed = r'''\ProvidesPackage{framed}[2023/04/10 dummy framed]

ewenvironment{framed}{}{}

ewenvironment{oframed}{}{}
\endinput
'''

write_sty('cleveref.sty', cleveref)
write_sty('framed.sty', framed)
