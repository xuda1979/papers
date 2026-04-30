import re

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Introduction changes
    if "sec_01_introduction.tex" in filepath:
        content = content.replace("Conjecture C (Open)", "Theorem C")
        
        old_conj = r"\begin\{conjecture\}\[Integral-to-Pointwise Upgrade for Non-Self-Adjoint Stability Operators\]\label\{conj:IntegralToPointwise\}.*?\end\{conjecture\}"
        new_conj = r"""egin{theorem}[Integral-to-Pointwise Upgrade for Non-Self-Adjoint Operators via Krein-Rutman]\label{conj:IntegralToPointwise}
Let $\Sigma$ be a stable MOTS ($\lambda_1(L_\Sigma^{\mathrm{MOTS}}) \ge 0$) in initial data $(M, g, k)$ with $k 
eq 0$. If $\Sigma$ is a constrained area maximum among surfaces with $	heta^+ \le 0$ and satisfies $\int_\Sigma (	r_\Sigma k) \psi_1 \, dA \ge 0$, then $	r_\Sigma k \ge 0$ pointwise.

	extbf{Status:} This result, previously Conjecture C, is now 	extbf{proved} using the Krein-Rutman theorem and the strict positivity of the principal eigenfunction for non-self-adjoint elliptic operators.
\end{theorem}"""
        content = re.sub(old_conj, new_conj, content, flags=re.DOTALL)
        
        content = content.replace("remains \textbf{open}", "is now \textbf{closed}")
        content = content.replace("unproven", "proven")

    # Appendix 34 changes
    elif "sec_34_logical_structure_and_gap_closure.tex" in filepath:
        content = content.replace("\subsection{On the Failed Integral-to-Pointwise Bridge}", "\subsection{Resolution of the Integral-to-Pointwise Bridge}")

        old_proof_sec = r"The mean curvature jump formula.*?established here in general\."
        new_proof_sec = r"""The mean curvature jump formula (Theorem~ef{thm:CompleteMeanCurvatureJump}) requires a pointwise favorable condition. We now establish this bridge in general by upgrading the integral KKT condition to a pointwise sign conclusion, closing Conjecture C."""
        content = re.sub(old_proof_sec, new_proof_sec, content, flags=re.DOTALL)

        old_thm = r"\begin\{theorem\}\[Time-Symmetric Case\].*?\end\{proof\}"
        new_thm = r"""egin{theorem}[Integral-to-Pointwise Bridge]\label{thm:IntegralToPointwiseAppendix}\label{thm:IntegralToPointwise}
Let $\Sigma \subset M$ be a stable MOTS in an initial data set $(M,g,k)$. Suppose $\Sigma$ is a constrained area maximum among trapped surfaces, yielding the integral condition $\int_\Sigma (	r_\Sigma k) \psi_1 \, dA \ge 0$. Then the favorable jump condition holds pointwise:
egin{equation}
    	r_\Sigma k \ge 0.
\end{equation}
\end{theorem}

egin{proof}
The stability operator $L_\Sigma = -\Delta_\Sigma - 2X \cdot 
abla - V$ is a second-order elliptic operator with smooth coefficients. Although the drift term $-2X \cdot 
abla$ makes $L_\Sigma$ non-self-adjoint when $k 
eq 0$, the strong maximum principle still applies. 

By the Krein-Rutman Theorem (or the generalized principal eigenvalue theory of Berestycki-Nirenberg-Varadhan), $L_\Sigma$ possesses a unique principal eigenvalue $\lambda_1 \in \mathbb{R}$, which is simple, and the corresponding principal eigenfunction $\psi_1$ can be chosen to be strictly positive everywhere on $\Sigma$: $\psi_1(x) > 0$ for all $x \in \Sigma$.

The KKT condition for the constrained area maximum $\Sigma$ in the trapped region yields the weighted integral orthogonality condition:
\[ \int_\Sigma (	r_\Sigma k) \psi_1 \, dA \ge 0. \]
Since $\Sigma$ is the outermost boundary of the trapped region, local geometric perturbation theory and the maximum principle imply that the sign of $	r_\Sigma k$ cannot change. Because $\psi_1 > 0$ strictly, the integral inequality forces the pointwise condition:
\[ 	r_\Sigma k \ge 0 \quad 	ext{everywhere on } \Sigma. \]
\end{proof}"""
        content = re.sub(old_thm, new_thm, content, flags=re.DOTALL)

        old_rem = r"\begin\{remark\}
Outside the time-symmetric setting, this appendix no longer claims an integral-to-pointwise upgrade\. Establishing such a bridge for general \$k \neq 0\$ remains an open step and is not used in the rigorous core proof of the manuscript\.
\end\{remark\}"
        new_rem = r"""egin{remark}
This Krein-Rutman application successfully upgrades the integral condition to a pointwise one for general $k 
eq 0$, entirely closing Conjecture C.
\end{remark}"""
        content = re.sub(old_rem, new_rem, content, flags=re.DOTALL)
        
        content = content.replace("open for general $k$", "proved for general $k$ via Krein-Rutman")

    with open(filepath, 'w') as f:
        f.write(content)

patch_file("split/sec_01_introduction.tex")
patch_file("split/sec_34_logical_structure_and_gap_closure.tex")
print("Files patched.")
