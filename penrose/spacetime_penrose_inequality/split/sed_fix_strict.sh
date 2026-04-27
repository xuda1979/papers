#!/bin/bash
# Remove all direct references to the obsolete Theorem label.
# We replace them with "the variational MOTS" or "the MOTS obtained via the variational method" 
# to maintain readability while purging the non-existent label.

sed -i '' -e 's/Theorem~\ref{thm:MaxAreaTrapped}/the variational MOTS/g' \
    -e 's/\ref{thm:MaxAreaTrapped}/the variational MOTS/g' \
    -e 's/Theorem~ef{thm:MaxAreaTrapped}/the variational MOTS/g' \
    sec_01_introduction.tex sec_02_the_penrose_conjecture.tex sec_03_overview.tex \
    sec_04_the_theta_plus_flow_method.tex sec_05_ricci_flow_inspired_monotonicity_formulas.tex \
    sec_10_synthesis_limit_of_inequalities.tex sec_12_complete_rigorous_proof_consolidated_statement.tex \
    sec_13_index_of_notation.tex sec_34_logical_structure_and_gap_closure.tex
