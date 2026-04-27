with open('sec_34_logical_structure_and_gap_closure.tex', 'r') as f:
    lines = f.read().splitlines()
    
out = []
env_stack = []

for line in lines:
    if r'egin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}' in line:
        line = line.replace(r'egin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}', r'egin{theorem}[Pointwise Sign Control via Weak Test Functions] \label{thm:GapClosed}')
    if r'egin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}' in line:
        line = line.replace(r'egin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}', r'egin{proposition}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}')
    if r'egin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}' in line:
        line = line.replace(r'egin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}', r'egin{proposition}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}')
    if r'egin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}' in line:
        line = line.replace(r'egin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}', r'egin{proposition}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}')
        
    line = line.replace(r'Open Problem~ef{thm:GapClosed}', r'Theorem~ef{thm:GapClosed}')

    if r'egin{theorem}' in line:
        env_stack.append('theorem')
    elif r'egin{proposition}' in line:
        env_stack.append('proposition')
    elif r'egin{lemma}' in line:
        env_stack.append('lemma')

    if env_stack and r'\end{openproblem}' in line:
        curr = env_stack.pop()
        line = line.replace(r'\end{openproblem}', r'\end{' + curr + '}')
    elif env_stack and r'\end{' + env_stack[-1] + '}' in line:
        env_stack.pop()
        
    out.append(line)

with open('sec_34_logical_structure_and_gap_closure.tex', 'w') as f:
    f.write('
'.join(out) + '
')
