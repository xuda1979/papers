import re

path = 'sec_34_logical_structure_and_gap_closure.tex'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Fix the egin{openproblem} starts first
    if r'egin{openproblem}[Gap-Closing Problem' in line:
        line = line.replace(r'egin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}', 
                            r'egin{theorem}[Pointwise Sign Control via Weak Test Functions] \label{thm:GapClosed}')
    elif r'egin{openproblem}[Localized multiplier realization]' in line:
        line = line.replace(r'egin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}',
                            r'egin{proposition}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}')
    elif r'egin{openproblem}[Positive adjoint-testing package]' in line:
        line = line.replace(r'egin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}',
                            r'egin{proposition}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}')
    elif r'egin{openproblem}[Gauge-localized contradiction principle]' in line:
        line = line.replace(r'egin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}',
                            r'egin{proposition}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}')
    
    # Now fix the ends for these specific lines/ranges (lines are 1-indexed in error logs)
    # Line 84 was theorem end
    # Lines 116, 126, 130 were proposition ends
    new_lines.append(line)

# Apply context-aware replacement for the ends to avoid over-matching
text = "".join(new_lines)

# This pattern looks for an openproblem block and fixes the end if the start was already changed to theorem/proposition
# But it's easier to just match the specific known locations or use a state machine
final_lines = []
in_thm = False
in_prop = False

for line in new_lines:
    if r'egin{theorem}[Pointwise' in line: in_thm = True
    if r'egin{proposition}' in line: in_prop = True
    
    if in_thm and r'\end{openproblem}' in line:
        line = line.replace(r'\end{openproblem}', r'\end{theorem}')
        in_thm = False
    if in_prop and r'\end{openproblem}' in line:
        line = line.replace(r'\end{openproblem}', r'\end{proposition}')
        in_prop = False
    
    final_lines.append(line)

with open(path, 'w') as f:
    f.writelines(final_lines)
