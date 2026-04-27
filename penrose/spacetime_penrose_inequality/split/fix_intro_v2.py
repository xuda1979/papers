import re
import os

path = '/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_01_introduction.tex'

with open(path, 'rb') as f:
    content = f.read()

# Fix common corrupted sequences specifically in this file
content = content.replace(b'
& Thm.~\r
', b' & Thm.~\r
') # just as a placeholder
content = content.replace(b'\r
ef{', b'\ref{')
content = content.replace(b'\r
', b'') # This is likely where it went wrong - a stray  before a 
 that then got interpreted as a line break by editors but was meant to be ef

# Let's target the exact byte sequence seen in the output for the table
# \item 	extbf{Theorem A (Stable MOTS):} ...
content = content.replace(b'\item \textbf{Theorem A (Stable MOTS):} Penrose inequality for \emph{outermost stable} MOTS --- proved without cosmic censorship or symmetry.
& Thm.~\ref{thm:penroseinitial}', 
                          b'Theorem A (Stable MOTS) & Thm.~\ref{thm:penroseinitial}')

# Final pass on any remaining ef{
content = content.replace(b'ef{', b'\ref{')
# But wait, ef{ was likely originally ef{. If we have ef{, it might be the result of  being removed from ef{.
# Wait, if  was backslash-r, and we had ef, it might be ef.

# Let's just fix the table block by finding the start and end and replacing the whole thing.
start_marker = b'\begin{center}
\begin{tabular}{|l|l|l|}'
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

with open(path, 'wb') as f:
    f.write(content)
