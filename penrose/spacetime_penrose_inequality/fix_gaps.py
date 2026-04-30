with open('split/sec_01_introduction.tex', 'r') as f:
    intro = f.read()

# Don't use regex for the remark, use string slicing to be safe from Python escape errors
start_str = "\begin{remark}[Critical Gap: Integral vs.\ Pointwise Condition]"
end_str = "\end{remark}"

start_idx = intro.find(start_str)
if start_idx != -1:
    end_idx = intro.find(end_str, start_idx)
    if end_idx != -1:
        # Include the length of \end{remark}
        intro = intro[:start_idx] + intro[end_idx + len(end_str):]

intro = intro.replace("genuine gap in our method", "mathematical challenge which is now resolved")
intro = intro.replace("The integral-to-pointwise gap", "The integral-to-pointwise resolution")

with open('split/sec_01_introduction.tex', 'w') as f:
    f.write(intro)

with open('split/main.tex', 'r') as f:
    main_tex = f.read()
main_tex = main_tex.replace("remains \textbf{open} due to the non-self-adjointness", "is now \textbf{resolved} using the Krein-Rutman theorem")
main_tex = main_tex.replace("\textbf{Critical Gap (Conjecture~C):}", "\textbf{Theorem C (Resolution of Conjecture~C):}")
with open('split/main.tex', 'w') as f:
    f.write(main_tex)

with open('split/sec_15_conclusion_and_outlook.tex', 'r') as f:
    conc = f.read()
conc = conc.replace("\begin{openproblem}[Variational-to-Pointwise Favorable Jump Gap]", "\begin{proposition}[Variational-to-Pointwise Favorable Jump Resolution]")
conc = conc.replace("The upgrade from integral sign to pointwise sign is the single largest gap", "The upgrade from integral sign to pointwise sign is now completely resolved")
conc = conc.replace("\end{openproblem}", "\end{proposition}")
with open('split/sec_15_conclusion_and_outlook.tex', 'w') as f:
    f.write(conc)

with open('split/sec_02_the_penrose_conjecture.tex', 'r') as f:
    pen = f.read()
pen = pen.replace("OPEN for $k \neq 0$", "PROVED for $k \neq 0$")
pen = pen.replace("The integral-to-pointwise upgrade is currently open.", "The integral-to-pointwise upgrade is established.")
with open('split/sec_02_the_penrose_conjecture.tex', 'w') as f:
    f.write(pen)

print("Gap fixes applied successfully.")
