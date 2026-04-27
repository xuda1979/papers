import os

path = 'sec_15_conclusion_and_outlook.tex'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(r''' egin{openproblem}[Rigorous Compactness for Maximum Area Surfaces]
Establish the compactness conditions (C2) and (C3) required for the existence of a Maximum Area Trapped Surface in the general case. Specifically:
egin{itemize}
    \item 	extbf{(C2) No loss of area to infinity:} Prove that a sequence of trapped surfaces cannot drift to infinity while maintaining large area.
    \item 	extbf{(C3) No loss of area to small scales:} Prove that area cannot concentrate into small, high-curvature bubbles that vanish in the limit.
\end{itemize}
	extit{Difficulty:} Medium-High. Likely requires geometric measure theory techniques similar to those used in the existence of MOTS.
\end{proposition}''', r''' egin{openproblem}[Rigorous Compactness for Maximum Area Surfaces]
Establish the compactness conditions (C2) and (C3) required for the existence of a Maximum Area Trapped Surface in the general case. Specifically:
egin{itemize}
    \item 	extbf{(C2) No loss of area to infinity:} Prove that a sequence of trapped surfaces cannot drift to infinity while maintaining large area.
    \item 	extbf{(C3) No loss of area to small scales:} Prove that area cannot concentrate into small, high-curvature bubbles that vanish in the limit.
\end{itemize}
	extit{Difficulty:} Medium-High. Likely requires geometric measure theory techniques similar to those used in the existence of MOTS.
\end{openproblem}''')

with open(path, 'w') as f:
    f.write(content)
