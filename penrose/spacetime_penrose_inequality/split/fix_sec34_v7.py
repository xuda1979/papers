import re

path = 'sec_34_logical_structure_and_gap_closure.tex'
with open(path, 'r') as f:
    content = f.read()

# Normalize backslashes if any literal backspaces remain
content = content.replace('', '\')

# 1. thm:GapClosed -> Theorem
# Using regex with flexible whitespace to match the messy state
content = re.sub(r'\begin\{openproblem\}\[Gap-Closing Problem for Non-Self-Adjoint Stability Operators\]\s*\label\{thm:GapClosed\}',
                 r'\begin{theorem}[Pointwise Sign Control via Weak Test Functions] \label{thm:GapClosed}', content)

# 2. op:LocalizedMultiplierRealization -> Proposition
content = re.sub(r'\begin\{openproblem\}\[Localized multiplier realization\]\s*\label\{op:LocalizedMultiplierRealization\}',
                 r'\begin{proposition}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}', content)

# 3. op:PositiveAdjointPackage -> Proposition
content = re.sub(r'\begin\{openproblem\}\[Positive adjoint-testing package\]\s*\label\{op:PositiveAdjointPackage\}',
                 r'\begin{proposition}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}', content)

# 4. op:GaugeLocalizedContradiction -> Proposition
content = re.sub(r'\begin\{openproblem\}\[Gauge-localized contradiction principle\]\s*\label\{op:GaugeLocalizedContradiction\}',
                 r'\begin{proposition}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}', content)

# Fix mismatched ends specifically in sec_34
# We look for a theorem or proposition that is ended by an openproblem
# and fix the end to match.
def fix_mismatched_ends(text):
    # Fix Theorems
    text = re.sub(r'(\begin\{theorem\}(?:(?!\begin).)*?)\end\{openproblem\}', r'\1\end{theorem}', text, flags=re.DOTALL)
    # Fix Propositions
    text = re.sub(r'(\begin\{proposition\}(?:(?!\begin).)*?)\end\{openproblem\}', r'\1\end{proposition}', text, flags=re.DOTALL)
    return text

content = fix_mismatched_ends(content)

# Update the cross-reference
content = content.replace('Open Problem~\ref{thm:GapClosed}', 'Theorem~\ref{thm:GapClosed}')

with open(path, 'w') as f:
    f.write(content)
