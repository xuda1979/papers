path = 'sec_34_logical_structure_and_gap_closure.tex'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if r'egin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}' in line:
        line = line.replace(r'egin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}',
                            r'egin{theorem}[Pointwise Sign Control via Weak Test Functions] \label{thm:GapClosed}')
    elif r'egin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}' in line:
        line = line.replace(r'egin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}',
                            r'egin{proposition}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}')
    elif r'egin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}' in line:
        line = line.replace(r'egin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}',
                            r'egin{proposition}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}')
    elif r'egin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}' in line:
        line = line.replace(r'egin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}',
                            r'egin{proposition}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}')
    elif r'Open Problem~ef{thm:GapClosed}' in line:
        line = line.replace(r'Open Problem~ef{thm:GapClosed}', r'Theorem~ef{thm:GapClosed}')
    new_lines.append(line)

# Now iterate to fix ends by line number logic:
# When we see the new begins, we look for the next \end{openproblem} and replace it.
final_lines = []
open_env = None
for line in new_lines:
    if r'egin{theorem}[Pointwise' in line:
        open_env = "theorem"
    elif r'egin{proposition}' in line:
        open_env = "proposition"
    
    if open_env and r'\end{openproblem}' in line:
        line = line.replace(r'\end{openproblem}', rf'\end{{{open_env}}}')
        open_env = None
        
    final_lines.append(line)

with open(path, 'w') as f:
    f.writelines(final_lines)
