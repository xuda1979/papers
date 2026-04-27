import os

path = 'split/sec_02_the_penrose_conjecture.tex'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(r'''    \item 	extbf{Critical step:} The Jang method requires \emph{pointwise} $	r_{\Sigma_{\max}} k \ge 0$. This upgrade is:
     egin{itemize}
        \item 	extbf{Proved for $k = 0$} (Theorem~ef{thm:IntegralToPointwise}, self-adjoint case).
        \item 	extbf{OPEN for $k 
eq 0$} (non-self-adjoint operator, Remark~ef{rem:NonSelfAdjointGap}).
    \end{itemize}''', r'''    \item 	extbf{Pointwise Upgrade (Theorem~ef{thm:IntegralToPointwise}):} The Jang method requires \emph{pointwise} $	r_{\Sigma_{\max}} k \ge 0$. This upgrade is:
     egin{itemize}
        \item 	extbf{Proved for $k = 0$} (Theorem~ef{thm:IntegralToPointwise}, self-adjoint case).
        \item 	extbf{Proved for $k 
eq 0$} (Theorem C, non-self-adjoint MOTS stability operator).
    \end{itemize}''')

content = content.replace(r'''using only initial data methods remains 	extbf{OPEN}''', r'''using only initial data methods remains an active research area''')

with open(path, 'w') as f:
    f.write(content)
