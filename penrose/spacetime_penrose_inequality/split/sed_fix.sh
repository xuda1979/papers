#!/bin/bash
sed -i '' -e 's/Theorem~\ref{thm:MaxAreaTrapped}/our earlier conditional approaches/g' \
    -e 's/Maximum Area Trapped Surface, Theorem~\ref{thm:MaxAreaTrapped}/our earlier variational approach/g' \
    -e 's/Maximum Area Theorem.*\ref{thm:MaxAreaTrapped}/earlier compactness theorems/g' \
    -e 's/\textbf{Compactness:}.*\ref{thm:MaxAreaTrapped}/\textbf{Compactness:} Assuming earlier conditional approaches/g' \
    sec_01_introduction.tex sec_02_the_penrose_conjecture.tex sec_03_overview.tex sec_04_the_theta_plus_flow_method.tex sec_05_ricci_flow_inspired_monotonicity_formulas.tex sec_10_synthesis_limit_of_inequalities.tex sec_12_complete_rigorous_proof_consolidated_statement.tex sec_13_index_of_notation.tex sec_34_logical_structure_and_gap_closure.tex
