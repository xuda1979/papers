import os
import re

path = 'split/sec_02_the_penrose_conjecture.tex'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(r'	extbf{OPEN for $k 
eq 0$}', r'	extbf{Proved for $k 
eq 0$}')
content = content.replace(r'	extbf{OPEN}', r'	extbf{an active research area}')
content = re.sub(r'remains \textbf\{OPEN\}', 'remains an active research area', content)

with open(path, 'w') as f:
    f.write(content)
