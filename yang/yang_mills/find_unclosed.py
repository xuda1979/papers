#!/usr/bin/env python3
import re

content = open('yang_mills.tex', 'r', encoding='utf-8').readlines()
depth = 0
unclosed = []

for i, line in enumerate(content, 1):
    if '\\begin{theorem}' in line:
        depth += 1
        unclosed.append((i, line.strip()[:80]))
    # Check for both \end{theorem} and \end{theorem> (typo)
    if re.search(r'\\end\{theorem[}>]', line):
        depth -= 1
        if unclosed:
            unclosed.pop()

print(f'Unclosed theorems: {len(unclosed)}')
for l in unclosed:
    print(f'Line {l[0]}: {l[1]}')
