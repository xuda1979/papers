with open('framed.sty', 'w') as f:
    f.write(r'''\ProvidesPackage{framed}[2023/04/10 dummy framed]

ewenvironment{framed}{}{}

ewenvironment{oframed}{}{}
\endinput
''')
