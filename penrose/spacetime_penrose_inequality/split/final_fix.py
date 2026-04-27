import os

def clean_and_fix(filepath, mappings):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # Read binary to handle any byte issues
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Remove the backspace char \x08 which is appearing as ghost characters
    clean_data = data.replace(b'\x08', b'')
    
    # Decode to text
    text = clean_data.decode('utf-8', errors='ignore')
    
    # Fix broken egin (if 0x08 was between \ and begin)
    text = text.replace('egin{', '\begin{')
    
    # Apply specific content mappings
    for old, new in mappings.items():
        text = text.replace(old, new)
        
    # Logic to fix mismatched end tags based on current environment
    lines = text.splitlines()
    fixed_lines = []
    current_env = None
    for line in lines:
        if '\begin{theorem}' in line:
            current_env = 'theorem'
        elif '\begin{proposition}' in line:
            current_env = 'proposition'
        elif '\begin{lemma}' in line:
            current_env = 'lemma'
        
        if current_env and '\end{openproblem}' in line:
            line = line.replace('\end{openproblem}', f'\end{{{current_env}}}')
            current_env = None
        elif current_env and (f'\end{{{current_env}}}' in line):
            current_env = None
            
        fixed_lines.append(line)
        
    with open(filepath, 'w') as f:
        f.write('
'.join(fixed_lines) + '
')

# Mappings for sec_34
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

# Mappings for sec_36
sec36_map = {
    '\begin{openproblem}[Spectral transfer for non-self-adjoint operators] \label{thm:SpectralTransfer}':
    '\begin{theorem}[Spectral transfer for non-self-adjoint operators] \label{thm:SpectralTransfer}'
}

clean_and_fix('sec_34_logical_structure_and_gap_closure.tex', sec34_map)
clean_and_fix('sec_36_dispersive_estimates_and_spectral_transfer.tex', sec36_map)
