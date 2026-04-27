import re
import os

path = '/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_01_introduction.tex'

with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# find the table manually by indices
start_idx = text.find('\begin{center}
\begin{tabular}{|l|l|l|}')
end_idx = text.find('\end{tabular}
\end{center}')

if start_idx != -1 and end_idx != -1:
    end_idx += len('\end{tabular}
\end{center}')
    
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
    
    text = text[:start_idx] + new_table + text[end_idx:]
    print("Table replaced.")
else:
    print("Markers not found.")

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
