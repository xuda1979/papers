import re

bs = chr(92)
nl = chr(10)

with open("sec_34_logical_structure_and_gap_closure.tex", "r") as f:
    text = f.read()

if "label{subsec:SpectralGapClosure}" not in text:
    text = text.replace(
        bs + "begin{theorem}[Spectral Deformation and Pointwise Upgrade]",
        bs + "subsection{Spectral Gap Closure for Non-Self-Adjoint Operators}" + nl + bs + "label{subsec:SpectralGapClosure}" + nl + bs + "begin{theorem}[Spectral Deformation and Pointwise Upgrade]"
    )

with open("sec_34_logical_structure_and_gap_closure.tex", "w") as f:
    f.write(text)

with open("sec_01_introduction.tex", "r") as f:
    text_intro = f.read()

text_intro = text_intro.replace(bs + "ref{subsec:SpectralGapClosure} (Theorem~" + nl + bs + "ref{thm:GapClosed})", bs + "ref{subsec:SpectralGapClosure} (Theorem~" + bs + "ref{thm:GapClosed})")

with open("sec_01_introduction.tex", "w") as f:
    f.write(text_intro)
