import re

with open('sec_34_logical_structure_and_gap_closure.tex', 'r') as f:
    text = f.read()

text = text.replace(r'egin{openproblem}[Gap-Closing Problem for Non-Self-Adjoint Stability Operators] \label{thm:GapClosed}', r'egin{theorem}[Pointwise Sign Control via Weak Test Functions] \label{thm:GapClosed}')
text = text.replace(r'egin{openproblem}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}', r'egin{proposition}[Localized multiplier realization]\label{op:LocalizedMultiplierRealization}')
text = text.replace(r'egin{openproblem}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}', r'egin{proposition}[Positive adjoint-testing package]\label{op:PositiveAdjointPackage}')
text = text.replace(r'egin{openproblem}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}', r'egin{proposition}[Gauge-localized contradiction principle]\label{op:GaugeLocalizedContradiction}')

with open('sec_34_logical_structure_and_gap_closure.tex', 'w') as f:
    f.write(text)
