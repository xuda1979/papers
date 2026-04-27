import re

def fix_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # sec 34
    content = content.replace(r' egin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}', r'egin{theorem}[Pointwise Sign Control via Weak Test Functions] \label{thm:GapClosed}')
    content = content.replace(r' egin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}', r'egin{proposition}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}')
    content = content.replace(r' egin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}', r'egin{proposition}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}')
    content = content.replace(r' egin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}', r'egin{proposition}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}')
    content = content.replace(r'Open Problem~ef{thm:GapClosed}', r'Theorem~ef{thm:GapClosed}')

    # sec 36
    content = content.replace(r' egin{openproblem}[Spectral Transfer Principle]\label{thm:SpectralTransfer}', r'egin{theorem}[Spectral Transfer Principle]\label{thm:SpectralTransfer}')
    content = content.replace(r' egin{openproblem}[Spectral transfer for non-self-adjoint operators] \label{thm:SpectralTransfer}', r'egin{theorem}[Spectral transfer for non-self-adjoint operators] \label{thm:SpectralTransfer}')
    content = content.replace(r' egin{proof}', r'egin{proof}')
    content = content.replace(r' egin{theorem}', r'egin{theorem}')

    lines = content.split('
')
    env_stack = []
    out_lines = []

    for line in lines:
        if r'egin{theorem}' in line:
            env_stack.append('theorem')
        elif r'egin{proposition}' in line:
            env_stack.append('proposition')
        elif r'egin{lemma}' in line:
            env_stack.append('lemma')

        if r'\end{openproblem}' in line and env_stack:
            curr = env_stack.pop()
            line = line.replace(r'\end{openproblem}', r'\end{' + curr + '}')
        elif r'\end{' in line and env_stack and r'\end{' + env_stack[-1] + '}' in line:
            env_stack.pop()
            
        out_lines.append(line)

    with open(filename, 'w') as f:
        f.write('
'.join(out_lines))

try:
    fix_file('sec_34_logical_structure_and_gap_closure.tex')
except Exception as e:
    print(e)
    
try:
    fix_file('sec_36_dispersive_estimates_and_spectral_transfer.tex')
except Exception as e:
    print(e)
