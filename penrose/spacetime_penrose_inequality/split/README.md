# Split Paper Structure

This directory contains the split version of `paper_clean.tex`.

## Structure

- **main.tex** - Main document that includes all section files
- **sec_XX_*.tex** - Individual section files (35 sections total)

## Section List

1. sec_01_introduction.tex - Introduction
2. sec_02_the_penrose_conjecture.tex - The Penrose Conjecture
3. sec_03_overview.tex - Overview
4. sec_04_the_texorpdfstringtheta.tex - The θ⁺-Flow Method
5. sec_05_ricci_flow_inspired_monotonicity_formulas_for_the_.tex - Ricci Flow-Inspired Monotonicity Formulas
6. sec_06_a_boost_invariant_quasi_local_mass_approach_to_the.tex - A Boost-Invariant Quasi-Local Mass Approach
7. sec_07_the_texorpdfstringp.tex - The p-Harmonic Level Set Method (AMO Framework)
8. sec_08_the_generalized_jang_reduction_and_analytical_obst.tex - The Generalized Jang Reduction
9. sec_09_analysis_of_the_singular_lichnerowicz_equation_and.tex - Analysis of the Singular Lichnerowicz Equation
10. sec_10_synthesis_limit_of_inequalities.tex - Synthesis: Limit of Inequalities
11. sec_11_rigidity_and_the_uniqueness_of_schwarzschild.tex - Rigidity and the Uniqueness of Schwarzschild
12. sec_12_complete_rigorous_proof_consolidated_statement.tex - Complete Rigorous Proof
13. sec_13_index_of_notation.tex - Index of Notation
14. sec_14_new_research_directions_lorentzian_optimal_transpo.tex - New Research Directions
15. sec_15_conclusion_and_outlook.tex - Conclusion and Outlook
16. sec_16_global_lipschitz_structure_of_the_jang_metric.tex - Global Lipschitz Structure of the Jang Metric
17. sec_17_geometric_measure_theory_analysis_of_the_smoothing.tex - Geometric Measure Theory Analysis
18. sec_18_spectral_positivity_and_removability_of_singularit.tex - Spectral Positivity and Removability
19. sec_19_capacity_of_singularities_and_flux_estimates.tex - Capacity of Singularities
20. sec_20_vanishing_flux_at_tips_correction.tex - Vanishing Flux at Tips
21. sec_21_distributional_identities_and_the_bochner_formula.tex - Distributional Identities and the Bochner Formula
22. sec_22_lockhart_mcowen_fredholm_theory_on_manifolds_with_.tex - Lockhart-McOwen Fredholm Theory
23. sec_23_estimates_for_the_internal_corner_smoothing.tex - Estimates for the Internal Corner Smoothing
24. sec_24_derivation_of_the_bray_khuri_divergence_identity.tex - Derivation of the Bray-Khuri Divergence Identity
25. sec_25_rigorous_scalar_curvature_estimates_for_the_smooth.tex - Rigorous Scalar Curvature Estimates
26. sec_26_the_marginally_trapped_limit_and_flux_cancellation.tex - The Marginally Trapped Limit
27. sec_27_declarations.tex - Declarations
28. sec_28_mosco_convergence_of_texorpdfstringp.tex - Mosco Convergence of p-Energies
29. sec_29_distributional_bochner_identity_with_measure_value.tex - Distributional Bochner Identity
30. sec_30_weak_inverse_mean_curvature_flow_and_hawking_mass_.tex - Weak Inverse Mean Curvature Flow
31. sec_31_optimal_transport_identification_of_adm_mass.tex - Optimal Transport Identification of ADM Mass
32. sec_32_worked_example_schwarzschild_initial_data.tex - Worked Example: Schwarzschild Initial Data
33. sec_33_complete_rigorous_mathematical_derivations.tex - Complete Rigorous Mathematical Derivations
34. sec_34_logical_structure_and_gap_closure.tex - Logical Structure and Gap Closure
35. sec_35_acknowledgments.tex - Acknowledgments

## How to Compile

### Using pdflatex
```bash
cd split
pdflatex main.tex
bibtex main     # if bibliography is needed
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk (recommended)
```bash
cd split
latexmk -pdf main.tex
```

## Editing

To edit a specific section:
1. Open the corresponding `sec_XX_*.tex` file
2. Make your changes
3. Recompile `main.tex`

## Notes

- Each section file contains only the section content (from `\section{...}` to the next section)
- The preamble (packages, macros, etc.) is in `main.tex`
- All section files are automatically included via `\input{...}` commands in `main.tex`
- Cross-references and labels work across all files
- The bibliography is included in the last section file (sec_35_acknowledgments.tex)

## Original File

The original file `paper_clean.tex` remains in the parent directory unchanged.

## Script

The split was created using `split_paper.ps1` (PowerShell script in parent directory).
