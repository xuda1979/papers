import re
import os

path = '/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_01_introduction.tex'

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# find the table manually by indices using regex, escaping correctly
match = re.search(r'\begin\{center\}.*?\begin\{tabular\}.*?\end\{tabular\}.*?\end\{center\}', text, re.DOTALL)

if match:
    new_table = """\begin{center}
\begin{tabular}{|l|l|l|}
\hline
\textbf{Result} & \textbf{Statement} & \textbf{Location} \\
\hline
Theorem A (Stable MOTS) & Thm.~\ref{thm:penroseinitial} & Sec.~\ref{sec:penrose_conjecture} (Summary form) \\
& Thm.~\ref{thm:SPI} & Sec.~\ref{sec:Synthesis} (Full proof) \\
\hline
Theorem B (Conditional) & Thm.~\ref{thm:intro-conditional} & Sec.~\ref{sec:intro} (Introduction) \\
& Thm.~\ref{thm:MainTheorem} & Sec.~\ref{sec:intro} (Detailed form) \\
& Thm.~\ref{thm:CompleteProof} & Sec.~\ref{sec:Consolidated} (Consolidated) \\
\hline
Conjecture C (Open for $k\neq 0$) & Conj.~\ref{conj:IntegralToPointwise} & Sec.~\ref{sec:intro} \\
\hline
\end{tabular}
\end{center}"""
    
    text = text[:match.start()] + new_table + text[match.end():]
    print("Table replaced.")
else:
    print("Markers not found.")

# Let's fix the other missing $ inserted error and undefined control sequences in this file
text = text.replace('\eqr
', '\eqref')
text = text.replace('\eqr

', '\eqref')
text = text.replace('\eqr
', '\eqref')
text = re.sub(r'\eqr\s*ef', r'\eqref', text)
text = re.sub(r'\eqr\s*\ef', r'\eqref', text)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
