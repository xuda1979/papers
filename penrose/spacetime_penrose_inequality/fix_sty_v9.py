import os

cleveref_content = "\ProvidesPackage{cleveref}[2023/04/10 dummy cleveref]
" \
"\DeclareOption*{}\ProcessOptions
" \
"\RequirePackage{hyperref}
" \
"\newcommand{\cref}[1]{\ref{#1}}
" \
"\newcommand{\Cref}[1]{\ref{#1}}
" \
"\newcommand{\cpageref}[1]{\pageref{#1}}
" \
"\newcommand{\Cpageref}[1]{\pageref{#1}}
" \
"\endinput
"

framed_content = "\ProvidesPackage{framed}[2023/04/10 dummy framed]
" \
"\newenvironment{framed}{}{}
" \
"\newenvironment{oframed}{}{}
" \
"\endinput
"

for d in ['.', 'split']:
    with open(os.path.join(d, 'cleveref.sty'), 'w', encoding='ascii') as f:
        f.write(cleveref_content)
    with open(os.path.join(d, 'framed.sty'), 'w', encoding='ascii') as f:
        f.write(framed_content)
