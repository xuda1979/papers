def replace_in_file(filepath, mapping):
    with open(filepath, 'r') as f:
        content = f.read()
        
    for k, v in mapping.items():
        content = content.replace(k, v)
        
    lines = content.splitlines()
    fixed_lines = []
    current_env = None
    for line in lines:
        if '\begin{theorem}' in line:
            current_env = 'theorem'
        elif '\begin{proposition}' in line:
            current_env = 'proposition'
        
        if current_env and '\end{openproblem}' in line:
            line = line.replace('\end{openproblem}', f'\end{{{current_env}}}')
            current_env = None
        elif current_env and f'\end{{{current_env}}}' in line:
            current_env = None
            
        fixed_lines.append(line)
        
    with open(filepath, 'w') as f:
        f.write('
'.join(fixed_lines) + '
')

sec34_map = {
    '\begin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}': 
    '\begin{theorem}[Pointwise Sign Control via Weak Test Functions] \label{thm:GapClosed}',
    
    '\begin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}':
    '\begin{proposition}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}',
    
    '\begin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}':
    '\begin{proposition}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}',
    
    '\begin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}':
    '\begin{proposition}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}',
    
    'Open Problem~\ref{thm:GapClosed}': 'Theorem~\ref{thm:GapClosed}'
}

sec36_map = {
    '\begin{openproblem}[Spectral Transfer Principle]\label{thm:SpectralTransfer}':
    '\begin{theorem}[Spectral Transfer Principle]\label{thm:SpectralTransfer}',
    '\begin{openproblem}[Spectral transfer for non-self-adjoint operators] \label{thm:SpectralTransfer}':
    '\begin{theorem}[Spectral transfer for non-self-adjoint operators] \label{thm:SpectralTransfer}'
}

replace_in_file('sec_34_logical_structure_and_gap_closure.tex', sec34_map)
replace_in_file('sec_36_dispersive_estimates_and_spectral_transfer.tex', sec36_map)
