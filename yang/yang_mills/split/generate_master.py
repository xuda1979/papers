import os

def get_files(prefix):
    files = [f for f in os.listdir('.') if f.startswith(prefix) and f.endswith('.tex')]
    files.sort()
    return files

sections = get_files('sec')
appendices = get_files('app')

# Filter sections to avoid duplicates, preferring '_new' or '_revised' or unique numbers
# This is a heuristic. For now, I'll just include all unique section NUMBERS, picking the longest filename for each number as it likely indicates "revised" or "new" or "rigorous".
# Actually, let's just include everything that looks like a distinct chapter.

# Better strategy: Group by number.
sec_dict = {}
for f in sections:
    # Extract number
    parts = f.split('_')
    if len(parts) > 0 and parts[0].startswith('sec') and parts[0][3:].isdigit():
        num = int(parts[0][3:])
        if num not in sec_dict:
            sec_dict[num] = []
        sec_dict[num].append(f)

# For each number, pick the best file.
# Preference: _new > _revised > _rigorous > original
final_sections = []
for num in sorted(sec_dict.keys()):
    candidates = sec_dict[num]
    chosen = candidates[0]
    for pref in ['_new', '_revised', '_rigorous']:
        for c in candidates:
            if pref in c:
                chosen = c
                break
        if chosen != candidates[0]:
            break
    final_sections.append(chosen)

# Appendices: Just include all of them, sorted numerically.
app_dict = {}
for f in appendices:
    # Extract number
    # Some are app29, some app100.
    # Some are app_honest...
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
    # For appendices, we might want to include all variants if they are different topics, 
    # but usually they are revisions. Let's pick the longest name, assuming it's more descriptive/newer.
    # Or just include all of them if they seem distinct.
    # The user wants the "900 page" paper. This implies including EVERYTHING.
    # But including duplicates (revised vs original) is bad.
    # Let's pick the one with 'complete' or 'rigorous' or 'new' if available.
    candidates = app_dict[num]
    chosen = candidates[0]
    for pref in ['_complete', '_rigorous', '_new', '_revised']:
        for c in candidates:
            if pref in c:
                chosen = c
                break
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

print("Created yang_mills_complete.tex")
