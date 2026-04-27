import re

with open('sec_34_logical_structure_and_gap_closure.tex', 'r') as f:
    content = f.read()

content = content.replace('int_Sigma', r'\int_\Sigma')
content = content.replace('r_Sigma', r'	r_\Sigma')
content = content.replace('ge 0', r'\ge 0')
content = content.replace('psi_1', r'\psi_1')
content = content.replace(', dA', r'\, dA')
content = content.replace('		extbf', r'	extbf')
content = content.replace('	extbf', r'	extbf')
content = content.replace('emph{', r'\emph{')
content = content.replace('Sigma', r'\Sigma')

with open('sec_34_logical_structure_and_gap_closure.tex', 'w') as f:
    f.write(content)

