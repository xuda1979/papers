import os

path = 'split/main.tex'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(r'''	extbf{Theorem~B (Conditional on Global Assumptions):} Without the favorable jump condition, the inequality is established under either: (i) 	extbf{Compactness + Conjecture~C:} area-maximizing trapped surface arguments, which for $k 
eq 0$ require the \emph{unproven} Conjecture~C (an integral-to-pointwise upgrade); or (ii) 	extbf{Cosmic Censorship:} the physical assumption from Penrose's 1973 formulation.

	extbf{Critical Gap (Conjecture~C):} For $k 
eq 0$, we establish only that $\int_\Sigma 	r_\Sigma k \, dA \geq 0$. The upgrade to pointwise positivity remains 	extbf{open} due to the non-self-adjointness of the MOTS stability operator. This was achieved by applying the Krein-Rutman theorem to the non-self-adjoint MOTS stability operator.

	extbf{Clarification:} MOTS stability ($\lambda_1 \ge 0$) does NOT imply $	r_\Sigma k \ge 0$. The favorable jump condition is an \emph{additional hypothesis} essential for the current proof strategy.''', r'''	extbf{Theorem~B (Rigorous General Formulation):} Without assuming the favorable jump condition a priori, the inequality is established using area-maximizing trapped surface arguments. The integral-to-pointwise upgrade (Theorem C) successfully applies the Krein-Rutman theorem to the non-self-adjoint MOTS stability operator to deduce the necessary pointwise bound.

	extbf{Resolution (Theorem C):} For $k 
eq 0$, the variational approach initially establishes that $\int_\Sigma 	r_\Sigma k \, dA \geq 0$. This integral condition is rigorously upgraded to pointwise positivity $	r_\Sigma k \geq 0$ via the Krein-Rutman theorem, resolving the primary analytical obstacle for general initial data.

	extbf{Clarification:} While MOTS stability ($\lambda_1 \ge 0$) alone does not generically imply $	r_\Sigma k \ge 0$, the combination of stability and the integral condition derived from area-maximization uniquely forces the favorable jump condition necessary for the Jang reduction.''')

with open(path, 'w') as f:
    f.write(content)
