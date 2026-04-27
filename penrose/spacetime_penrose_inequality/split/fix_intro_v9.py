import re
import os

path = '/Users/daxu/papers/penrose/spacetime_penrose_inequality/split/sec_01_introduction.tex'

with open(path, 'rb') as f:
    data = f.read()

# Pattern for the corrupted table. Using byte literals to avoid escape issues.
# Looking for egin{center} ... egin{tabular} ... \end{tabular} ... \end{center}
# Note:  is x08 in many of these files.
pattern = re.compile(b'\\begin\{center\}.*?\\begin\{tabular\}.*?\\end\{tabular\}.*?\\end\{center\}', re.DOTALL)

# Also handle the x08 variant
pattern_x08 = re.compile(b'\x08egin\{center\}.*?\x08egin\{tabular\}.*?\x08nd\{tabular\}.*?\x08nd\{center\}', re.DOTALL)

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

match = pattern.search(data)
if not match:
    match = pattern_x08.search(data)

if match:
    data = data[:match.start()] + new_table + data[match.end():]
    print("Table replaced.")
else:
    print("Table markers not found in binary search.")

# Fix fragmented ef and \eqref
# 
ef -> ef
data = data.replace(b'\r\nef', b'\\ref')
data = data.replace(b'\r\neqref', b'\\eqref')
# also handle literal 
 in the middle of a command
data = data.replace(b'\eq\r\nref', b'\\eqref')

with open(path, 'wb') as f:
    f.write(data)
