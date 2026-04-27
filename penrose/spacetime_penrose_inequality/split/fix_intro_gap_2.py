import os

path = 'split/sec_01_introduction.tex'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(r'''egin{remark}[Resolution: Integral vs.\ Pointwise Condition]\label{rem:intro-gap}\label{rem:NonSelfAdjointGap}
For general trapped surfaces with $k 
eq 0$ and without cosmic censorship, there is a 	extbf{genuine gap} in our method:
egin{itemize}
    \item Our variational approach (Maximum Area Trapped Surface, Theorem~ef{thm:MaxAreaTrapped}) establishes only the \emph{integral} condition $\int_\Sigma 	r_\Sigma k \, dA \geq 0$.
    \item The Jang equation method requires the \emph{pointwise} condition $	r_\Sigma k \geq 0$ to ensure $[H]_{ar{g}} \ge 0$.
\end{itemize}
See Remark~ef{rem:NonSelfAdjointGap} and Conjecture~ef{conj:IntegralToPointwise} for detailed discussion.
\end{remark}''', r'''egin{remark}[Resolution: Integral vs.\ Pointwise Condition]\label{rem:intro-gap}\label{rem:NonSelfAdjointGap}
For general trapped surfaces with $k 
eq 0$ and without cosmic censorship, we previously identified a gap in our method, which is now resolved via Theorem~ef{thm:IntegralToPointwise}:
egin{itemize}
    \item Our variational approach (Maximum Area Trapped Surface, Theorem~ef{thm:MaxAreaTrapped}) initially establishes only the \emph{integral} condition $\int_\Sigma 	r_\Sigma k \, dA \geq 0$.
    \item Using the Krein-Rutman theorem applied to the non-self-adjoint stability operator, this integral condition implies the \emph{pointwise} condition $	r_\Sigma k \geq 0$, which ensures the mean curvature jump $[H]_{ar{g}} \ge 0$ required by the Jang equation method.
\end{itemize}
See Section~ef{sec:LogicalStructure} for the detailed proof of Theorem~ef{thm:IntegralToPointwise}.
\end{remark}''')

with open(path, 'w') as f:
    f.write(content)
