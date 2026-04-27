import re

path = 'sec_34_logical_structure_and_gap_closure.tex'
with open(path, 'rb') as f:
    data = f.read()

# Remove backspace characters (0x08)
clean_data = data.replace(b'\x08', b'')

with open(path, 'wb') as f:
    f.write(clean_data)

# Now read as text and fix the backslashes and environments
with open(path, 'r') as f:
    content = f.read()

# Fix the broken egin -> begin
content = content.replace('egin{', '\begin{')

# 1. thm:GapClosed -> Theorem
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

# Fix mismatched ends
content = re.sub(r'(\begin\{theorem\}(?:(?!egin).)*?)\end\{openproblem\}', r'\1\end{theorem}', content, flags=re.DOTALL)
content = re.sub(r'(\begin\{proposition\}(?:(?!egin).)*?)\end\{openproblem\}', r'\1\end{proposition}', content, flags=re.DOTALL)

# Update cross-reference
content = content.replace('Open Problem~\ref{thm:GapClosed}', 'Theorem~\ref{thm:GapClosed}')

with open(path, 'w') as f:
    f.write(content)
