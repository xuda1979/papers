B=chr(92); NL=chr(10); SQ=chr(39)
txt = open("split/app213_rigorous_gap_and_final_audit.tex").read()
s = txt.find("begin{theorem}[Dobrushin--Shlosman")
e = txt.find("end{proof}", s)
end_pos = e + 10
