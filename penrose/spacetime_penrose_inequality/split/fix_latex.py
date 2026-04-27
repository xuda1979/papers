with open('sec_36_dispersive_estimates_and_spectral_transfer.tex', 'r') as f:
    text = f.read()

# Fix the specific lines
text = text.replace('egin{', '\begin{')
text = text.replace('\bbbegin', '\begin')
text = text.replace('\bbegin', '\begin')

# Manually fix the equation references in Step 5
old_snippet = 'Substituting this into \eqr
ef{eq:GapClosureLinearizedEquation} and dividing by $\phi>0$ yields \eqr
ef{eq:GroundStateDriftIdentity}. Under \eqr
ef{eq:GradientDriftCondition},'
new_snippet = 'Substituting this into \eqref{eq:GapClosureLinearizedEquation} and dividing by $\phi>0$ yields \eqref{eq:GroundStateDriftIdentity}. Under \eqref{eq:GradientDriftCondition},'
text = text.replace(old_snippet, new_snippet)

old_snippet2 = 'so \eqr
ef{eq:GroundStateDriftIdentity} becomes \eqr
ef{eq:WeightedGapClosureGradientForm}.'
new_snippet2 = 'so \eqref{eq:GroundStateDriftIdentity} becomes \eqref{eq:WeightedGapClosureGradientForm}.'
text = text.replace(old_snippet2, new_snippet2)

# Global replacements
text = text.replace('ef{', 'ref{')
text = text.replace('ref{ref{', 'ref{')
text = text.replace('\eqref{ref{', '\eqref{')

with open('sec_36_dispersive_estimates_and_spectral_transfer.tex', 'w') as f:
    f.write(text)
