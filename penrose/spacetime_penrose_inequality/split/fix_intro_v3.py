import re
import os

path = '/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_01_introduction.tex'

with open(path, 'rb') as f:
    content = f.read()

start_marker = b'egin{center}
egin{tabular}{|l|l|l|}'
end_marker = b'\end{tabular}
\end{center}'

if start_marker in content and end_marker in content:
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker) + len(end_marker)
    
    new_table = b"""\begin{center}
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
    
    content = content[:start_idx] + new_table + content[end_idx:]
    print("Table replaced.")
else:
    print("Markers not found.")

with open(path, 'wb') as f:
    f.write(content)
