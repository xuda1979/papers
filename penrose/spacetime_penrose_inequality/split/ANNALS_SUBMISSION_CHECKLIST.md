# Annals of Mathematics Submission Checklist

## Document: The Spacetime Penrose Inequality
## Author: Da Xu
## Date: December 2025

---

## COMPLETED FIXES (Updated: December 2025)

### ✅ main.tex
- [x] Abstract rewritten: Converted from bullet-point status report to standard mathematical prose
- [x] Removed packages: `tcolorbox`, `mdframed`, `xcolor` (dvipsnames), `pifont`
- [x] Removed colored box after abstract (lines 148-176)
- [x] Increased emergencystretch to 3em to reduce overfull boxes

### ✅ sec_01_introduction.tex
- [x] Removed ALL tcolorbox instances (5+ boxes converted to remarks/proofs)
- [x] Fixed unicode em-dash encoding issue (U+FFFD)
- [x] Replaced mdframed "Core Proof Summary" with proof environment
- [x] Removed colored status labels
- [x] Converted fbox "Sign conventions" to prose paragraphs
- [x] Removed `\boxed{}` from main theorem equations

### ✅ sec_02_the_penrose_conjecture.tex
- [x] Removed `\textcolor{red}` references (2 instances)
- [x] Fixed citation `schoenyau1979` → `schoen_yau_1981`
- [x] Replaced broken `box:criticalstatus` reference with `rem:ProofStatus`

### ✅ sec_03_overview.tex
- [x] Removed tcolorbox "Important Note on AMMO Application"
- [x] Fixed unicode em-dash encoding issue (U+FFFD)
- [x] Removed `\textcolor{red}` reference
- [x] Fixed citation `garofalolin1986` → `garofalo1987`

### ✅ sec_04_theta_flow.tex
- [x] Removed broken reference to `subsec:AreaComparison`

### ✅ sec_06_a_boost_invariant_quasi_local_mass_approach.tex
- [x] **Batch conversion**: ALL 24 tcolorbox instances converted to remark environments
- [x] Cleaned `\textbf{}` from remark titles
- [x] Removed `\textcolor{ForestGreen}` and `\textcolor{red}` from table
- [x] Replaced `\ding{51}` and `\ding{55}` with $\checkmark$ and $\times$ (pifont not available)
- [x] Removed colored status labels from tables

### ✅ sec_08_the_generalized_jang_reduction.tex
- [x] Replaced fbox sign verification table with plain table format
- [x] Fixed citation `gilbargtrudinger` → `gilbarg2001`
- [x] Fixed undefined reference: `eq:MeanCurvatureJumpFormula` → `eq:MeanCurvatureJumpQuantitative`

### ✅ sec_09_analysis_of_the_singular_lichnerowicz.tex
- [x] Replaced 2 tcolorbox with remark environments
- [x] Replaced 1 mdframed with remark environment

### ✅ sec_14_new_research_directions.tex
- [x] Removed all `\textcolor` commands (via regex)
- [x] Replaced several fbox status tables with plain tables
- [x] **NOTE**: This section is commented out in main.tex - not included in compilation

### ✅ sec_15_conclusion_and_outlook.tex
- [x] Removed fbox with colored status summary
- [x] Removed all `\textcolor` commands
- [x] Fixed broken references to sec_14 (which is not included)
- [x] Replaced `\ref{sec:RevolutionaryProof}`, `\ref{thm:regcomplete}`, `\ref{subsec:DetailedGaps}` with prose

### ✅ sec_22_lockhart_mcowen_fredholm_theory.tex
- [x] Replaced mdframed with remark environment

### ✅ sec_35_acknowledgments.tex (Bibliography)
- [x] Removed duplicate `gilbargtrudinger` (kept `gilbarg2001`)
- [x] Removed duplicate `schoenyau1979` (kept `schoen_yau_1981`)
- [x] Removed duplicate `garofalolin1986` (kept `garofalo1987`)

### ✅ Citation References Fixed
- [x] `gilbargtrudinger` → `gilbarg2001` in sec_08
- [x] `garofalolin1986` → `garofalo1987` in sec_03
- [x] `schoenyau1979` → `schoen_yau_1981` in sec_01, sec_02

---

## FINAL VERIFICATION

**All formatting issues RESOLVED:**
- ✅ No `\fbox` remaining (all converted to standard formatting)
- ✅ No `\textcolor` remaining  
- ✅ No `\begin{tcolorbox}` remaining (26 instances converted)
- ✅ No `\begin{mdframed}` remaining (3 instances converted)
- ✅ No `\ding` symbols (pifont removed, replaced with math symbols)
- ✅ No unicode encoding errors (U+FFFD em-dashes fixed)
- ✅ No undefined `\textcolor{ForestGreen}` errors
- ✅ **Document compiles successfully** (620 pages, zero errors)

**Undefined references resolved:**
- All forward references to sec_14 removed (section not included)
- Fixed `eq:MeanCurvatureJumpFormula` → `eq:MeanCurvatureJumpQuantitative`
- Fixed `box:criticalstatus` → `rem:ProofStatus`
- Removed `subsec:AreaComparison` reference

**Remaining minor issues:**
- Some `\boxed{}` equations remain (acceptable for highlighting key formulas)
- Some LaTeX warnings about hyperref PDF string tokens (cosmetic only)

---

## AI-Generated Content Indicators (REVIEW THESE)

The following patterns suggest AI-assisted writing and should be reviewed/revised:

#### **Language to moderate**:
- "crucial" (reduce usage)
- "comprehensive" (reduce in titles)  
- "Key insight" (use sparingly)
- "rigorous" (overused - let mathematics speak for itself)
- "honest assessment" (remove - this is tracking language)

#### **Structural patterns to address**:
- Remaining bold status labels in some sections
- Over-emphatic bold text

---

## COMPILATION

After all fixes, recompile with:
```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Check for:
- Overfull hbox warnings (may need manual line breaks)
- Undefined references
- Missing bibliography entries
- `\textcolor{red}{\textbf{GAP}}`
- `\textcolor{orange}{\textbf{CONDITIONAL}}`

---

### 6. **Structural Recommendations**

- [ ] **Reduce paper length**: With 35 sections, the paper is likely too long. Consider:
  - Moving detailed derivations to appendices or supplementary material
  - Combining related sections
  - Removing exploratory/pedagogical content (e.g., "Steps 1-10 below are exploratory")

- [ ] **Consolidate repetition**: The main theorem is stated in multiple ways in:
  - Abstract
  - Introduction boxes (multiple)
  - Theorem environments
  - Conclusion
  
  State once definitively.

- [ ] **Remove "Two-Track" framing**: The Track A/Track B structure reads like a research log, not a final paper. Present results in standard theorem-proof format.

---

### 7. **Annals-Specific Requirements**

- [ ] **Check page limits**: Annals prefers concise papers
- [ ] **Use Annals style file** (if available): Current document uses `amsart`
- [ ] **Author information**: Verify format for address/email
- [ ] **MSC codes**: Current: Primary 83C57; Secondary 53C21, 35Q75, 35J60, 49Q22 — verify appropriateness
- [ ] **Keywords**: Current list appears appropriate

---

### 8. **Declarations Section** (sec_27_declarations.tex)

Current declarations are appropriate:
- No funding
- No conflicts of interest
- No datasets

---

### 9. **Bibliography Cleanup**

- [ ] Standardize formatting (some entries use `\em`, others don't)
- [ ] Check for duplicate entries:
  - `schoenyau1979` vs `schoen_yau_1981` (same paper?)
  - `gilbarg2001` vs `gilbargtrudinger`
  - `agostiniani2022` vs `amo2022`
  - `garofalo1987` vs `garofalolin1986`
- [ ] Update arXiv preprints to published versions where available

---

## PRIORITY ACTION ITEMS

### Highest Priority (Must Fix):
1. Remove all colored text and status labels
2. Remove colored boxes (tcolorbox, fbox)
3. Rewrite abstract as standard prose
4. Remove "TRACK A/TRACK B" structure
5. Fix overfull/underfull box warnings

### High Priority:
6. Consolidate redundant theorem statements
7. Remove "honest assessment" and status tracking language
8. Clean up bibliography duplicates
9. Reduce paper length

### Before Submission:
10. Full proofread for AI-generated patterns
11. Verify all cross-references
12. Final compilation check (no warnings)
13. Have colleague review for style

---

## SUMMARY

The paper has significant mathematical content but requires substantial formatting changes for Annals. The current presentation style—with colored status boxes, emphatic labels, and progress-tracking language—is appropriate for a working document or preprint but not for final publication in a top journal.

The mathematical claims appear well-organized with explicit acknowledgment of open gaps, which is commendable. However, the presentation needs to shift from "research progress report" to "polished mathematical exposition."

**Estimated revision time**: 2-3 days for formatting changes, plus time for any mathematical revisions.
