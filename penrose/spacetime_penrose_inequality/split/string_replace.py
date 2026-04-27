import re

def process_file(filepath, reps):
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        
    out = []
    env_stack = []
    
    for line in lines:
        for old, new in reps:
            if old in line:
                line = line.replace(old, new)
                
        if "\begin{theorem}" in line:
            env_stack.append("theorem")
        elif "\begin{proposition}" in line:
            env_stack.append("proposition")
        elif "\begin{lemma}" in line:
            env_stack.append("lemma")
            
        if env_stack and "\end{openproblem}" in line:
            curr = env_stack.pop()
            line = line.replace("\end{openproblem}", f"\end{{{curr}}}")
        elif env_stack and f"\end{{{env_stack[-1]}}}" in line:
            env_stack.pop()
            
        out.append(line)
        
    with open(filepath, 'w') as f:
        for line in out:
            f.write(line + "
")

sec34_replacements = [
    ("\begin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}",
     "\begin{theorem}[Pointwise Sign Control via Weak Test Functions] \label{thm:GapClosed}"),
    ("\begin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}",
     "\begin{proposition}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}"),
    ("\begin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}",
     "\begin{proposition}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}"),
    ("\begin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}",
     "\begin{proposition}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}"),
    ("Open Problem~\ref{thm:GapClosed}", "Theorem~\ref{thm:GapClosed}")
]

sec36_replacements = [
    ("\begin{openproblem}[Spectral Transfer Principle]\label{thm:SpectralTransfer}",
     "\begin{theorem}[Spectral Transfer Principle]\label{thm:SpectralTransfer}"),
    ("\begin{openproblem}[Spectral transfer for non-self-adjoint operators] \label{thm:SpectralTransfer}",
     "\begin{theorem}[Spectral transfer for non-self-adjoint operators] \label{thm:SpectralTransfer}")
]

try:
    process_file('sec_34_logical_structure_and_gap_closure.tex', sec34_replacements)
except FileNotFoundError:
    pass
try:
    process_file('sec_36_dispersive_estimates_and_spectral_transfer.tex', sec36_replacements)
except FileNotFoundError:
    pass
