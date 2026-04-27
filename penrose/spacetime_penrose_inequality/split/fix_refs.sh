#!/bin/bash

# Find and replace lingering references to thm:MaxAreaTrapped
sed -i '' -e 's/Theorem~\ref{thm:MaxAreaTrapped}/the variational MOTS approach/g' sec_01_introduction.tex sec_02_the_penrose_conjecture.tex sec_03_overview.tex sec_04_the_theta_plus_flow_method.tex sec_05_ricci_flow_inspired_monotonicity_formulas.tex sec_10_synthesis_limit_of_inequalities.tex sec_12_complete_rigorous_proof_consolidated_statement.tex sec_13_index_of_notation.tex

sed -i '' -e 's/Lemma~\ref{lem:VanishingMultiplier}/the earlier vanishing multiplier hypothesis/g' sec_33_complete_rigorous_mathematical_derivations.tex

