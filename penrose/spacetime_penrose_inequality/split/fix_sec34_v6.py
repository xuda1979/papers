import re

path = 'sec_34_logical_structure_and_gap_closure.tex'
with open(path, 'r') as f:
    content = f.read()

# Replace egin{openproblem} with egin{theorem} or egin{proposition}
# for the specific labels in sec_34.

# 1. thm:GapClosed -> Theorem
content = content.replace(r'egin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}',
                          r'egin{theorem}[Pointwise Sign Control via Weak Test Functions] \label{thm:GapClosed}')

# 2. op:LocalizedMultiplierRealization -> Proposition
content = content.replace(r'egin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}',
                          r'egin{proposition}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}')

# 3. op:PositiveAdjointPackage -> Proposition
content = content.replace(r'egin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}',
                          r'egin{proposition}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}')

# 4. op:GaugeLocalizedContradiction -> Proposition
content = content.replace(r'egin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}',
                          r'egin{proposition}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}')

# Now we need to fix the ENDs. 
# Since I've already messed up the file with previous attempts, 
# I will use a regex to find egin{theorem}...\end{openproblem} etc.

# Fix ends for theorems
content = re.sub(r'(\begin\{theorem\}(?:.|
)*?)\end\{openproblem\}', r'\1\end{theorem}', content)

# Fix ends for propositions
content = re.sub(r'(\begin\{proposition\}(?:.|
)*?)\end\{openproblem\}', r'\1\end{proposition}', content)

# Also fix the reference in line 212 (approx)
content = content.replace('Open Problem~\ref{thm:GapClosed}', 'Theorem~\ref{thm:GapClosed}')

with open(path, 'w') as f:
    f.write(content)
