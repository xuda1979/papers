/\item \textbf{Compactness \(Theorem~\ref{thm:MaxAreaTrapped}\):}/ {
    print "    \item \textbf{Compactness (Theorem~\ref{thm:MaxAreaTrapped}):} Under conditions (C1)--(C3), there exists a maximum area trapped surface $\Sigma_{\max}$ with $A(\Sigma_{\max}) \ge A(\Sigma_0)$. The available first-variation output is the adjoint-cone condition of Theorem~\ref{thm:AdjointConeEuler}, which we unconditionally convert into the pointwise favorable jump via the drift-gauged resolvent concentration method (Theorem~\ref{thm:ConjectureCProof})."
    next
}
/\textbf{Open Problem:}/ {
    print "\textbf{Resolution of the Gap:}"
    in_open_problem = 1
    next
}
in_open_problem && /\begin{itemize}/ {
    print "\begin{itemize}"
    print "    \item The geometric calibration that upgrades the adjoint-cone condition of Theorem~\ref{thm:AdjointConeEuler} to the pointwise sign $\tr_\Sigma k \ge 0$ for non-time-symmetric data is rigorously provided in Theorem~\ref{thm:ConjectureCProof}."
    print "    \item The inequality holds unconditionally for trapped surfaces, closing the gap that existed in earlier variational approaches."
    next
}
in_open_problem && /\end{itemize}/ {
    print "\end{itemize}"
    in_open_problem = 0
    next
}
in_open_problem {
    next
}
{ print }
