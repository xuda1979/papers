import os

def get_files(prefix):
    files = [f for f in os.listdir('.') if f.startswith(prefix) and f.endswith('.tex')]
    files.sort()
    return files

sections = get_files('sec')
appendices = get_files('app')

# Group sections by number
sec_dict = {}
for f in sections:
    # Extract number
    parts = f.split('_')
    if len(parts) > 0 and parts[0].startswith('sec') and parts[0][3:].isdigit():
        num = int(parts[0][3:])
        if num not in sec_dict:
            sec_dict[num] = []
        sec_dict[num].append(f)

# Select best section for each number
final_sections = []
for num in sorted(sec_dict.keys()):
    candidates = sec_dict[num]
    # Priority: _new > _revised > _rigorous > _complete > original
    # But also check for specific "definitive" or "comprehensive" names
    chosen = candidates[0]
    best_score = -1
    
    for c in candidates:
        score = 0
        if '_new' in c: score += 2
        if '_revised' in c: score += 1
        if '_rigorous' in c: score += 3
        if '_complete' in c: score += 2
        if '_definitive' in c: score += 4
        
        if score > best_score:
            best_score = score
            chosen = c
            
    final_sections.append(chosen)

# Appendices: Include ALL unique numbered appendices.
# If multiple files exist for the same number (e.g. app158_...), pick the best one.
app_dict = {}
for f in appendices:
    if f.startswith('app') and f[3].isdigit():
        # Find where digits end
        i = 3
        while i < len(f) and f[i].isdigit():
            i += 1
        num = int(f[3:i])
        if num not in app_dict:
            app_dict[num] = []
        app_dict[num].append(f)
    else:
        # app_honest...
        if 'misc' not in app_dict:
            app_dict['misc'] = []
        app_dict['misc'].append(f)

final_appendices = []
for num in sorted([k for k in app_dict.keys() if isinstance(k, int)]):
    candidates = app_dict[num]
    chosen = candidates[0]
    best_score = -1
    for c in candidates:
        score = 0
        if '_new' in c: score += 2
        if '_revised' in c: score += 1
        if '_rigorous' in c: score += 3
        if '_complete' in c: score += 2
        if '_definitive' in c: score += 4
        if '_response' in c: score += 5 # High priority for response to reviews? Maybe not.
        
        if score > best_score:
            best_score = score
            chosen = c
    final_appendices.append(chosen)

if 'misc' in app_dict:
    final_appendices.extend(app_dict['misc'])

# Generate LaTeX content
latex_content = r"""\documentclass[11pt,a4paper]{article}
\input{preamble}
\input{document-info}

\begin{document}
\maketitle
\begin{abstract}
\input{abstract_revised}
\end{abstract}
\tableofcontents
\newpage

\part{Main Text}
"""

for s in final_sections:
    latex_content += f"\\input{{{s}}}\n"

latex_content += r"""
\part{Appendices}
\appendix
"""

for a in final_appendices:
    latex_content += f"\\input{{{a}}}\n"

latex_content += r"\end{document}"

with open('yang_mills_complete.tex', 'w', encoding='utf-8') as f:
    f.write(latex_content)

print(f"Created yang_mills_complete.tex with {len(final_sections)} sections and {len(final_appendices)} appendices.")
