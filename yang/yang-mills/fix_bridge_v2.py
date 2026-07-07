
B = chr(92)
NL = chr(10)
SQ = chr(39)

with open('split/app213_rigorous_gap_and_final_audit.tex','r') as fh:
    text = fh.read()

s_start = B + 'begin{theorem}[Dobrushin--Poincar' + B + SQ + '{e} implies OS mass gap]'
s_end_str = 'giving $m(' + B + 'Lambda,' + B + 'beta)' + B + 'ge' + B + 'kappa(' + B + 'beta)$.' + NL + B + 'end{proof}'

idx_s = text.find(s_start)
idx_e = text.find(s_end_str, idx_s)
end_pos = idx_e + len(s_end_str)

print('FOUND: start=%d end=%d end_pos=%d' % (idx_s, idx_e, end_pos))
