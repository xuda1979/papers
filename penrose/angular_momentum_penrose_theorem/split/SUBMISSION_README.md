# Submission Guide for Peking Mathematical Journal

## Paper Information
- **Title:** The Angular Momentum Penrose Inequality
- **Author:** Da Xu
- **Affiliation:** China Mobile Research Institute, Beijing, China
- **Subject:** Mathematical General Relativity, Geometric Analysis

---

## Submission Files

### Option A: Using Springer svjour3 Class (Recommended)
Compile: `pmj_submission.tex`

Required files:
- `pmj_submission.tex` (main file)
- `svjour3.cls` (Springer class file)
- All section files (`sec01-introduction.tex`, etc.)
- All appendix files (`app01-amo-estimates.tex`, etc.)
- `bibliography.tex`

### Option B: Using Standard Article Class
Compile: `pmj_submission_article.tex`

This version uses standard LaTeX article class with Springer-compatible formatting.

---

## Compilation Instructions

```bash
# Compile with pdflatex (run twice for TOC)
pdflatex pmj_submission.tex
pdflatex pmj_submission.tex

# Or use latexmk for automatic compilation
latexmk -pdf pmj_submission.tex
```

---

## Submission Checklist

### Before Submission
- [ ] Verify all equations compile correctly
- [ ] Check all cross-references resolve
- [ ] Verify bibliography entries are complete
- [ ] Run spell-check on all .tex files
- [ ] Review against `checklist_avoid_ai_style.md`

### Required Components (Springer/PMJ)
- [x] Title and author information
- [x] Abstract (max 250 words for PMJ)
- [x] Keywords (separated by `\and`)
- [x] MSC 2020 classification codes
- [x] Acknowledgments section
- [x] Conflict of Interest statement
- [x] Data Availability statement
- [x] Complete bibliography

### Formatting Requirements
- [x] 11pt font size (smallextended option)
- [x] A4 paper size
- [x] 2.5cm margins
- [x] Numbered sections and theorems
- [x] Numbered equations (where referenced)
- [x] References in numbered format

---

## Submission Portal

**Peking Mathematical Journal** uses Editorial Manager:
- URL: https://www.editorialmanager.com/pema/
- Or via Springer submission system

### Files to Upload
1. **Main manuscript:** `pmj_submission.pdf` (compiled PDF)
2. **LaTeX source:** All `.tex` files as a ZIP archive
3. **Cover letter:** (prepare separately)

---

## Cover Letter Template

```
Dear Editors,

I am pleased to submit the manuscript entitled "The Angular Momentum Penrose 
Inequality" for consideration in Peking Mathematical Journal.

This paper establishes the Penrose inequality with angular momentum for 
asymptotically flat, axisymmetric vacuum initial data. The main result proves:

    M_ADM ≥ √(A/16π + 4πJ²/A)

with equality if and only if the data arises from a slice of Kerr spacetime.

The paper combines techniques from the Jang equation approach (Bray-Khuri, 
Han-Khuri) with the p-harmonic level set method (Agostiniani-Mazzieri-Oronzio), 
introducing a new angular-momentum-corrected Hawking mass that is monotone 
along the flow.

This work addresses a longstanding open problem in mathematical general 
relativity and represents the first complete proof of the Penrose inequality 
that accounts for angular momentum.

The manuscript has not been published elsewhere and is not under consideration 
at any other journal.

Thank you for your consideration.

Sincerely,
Da Xu
China Mobile Research Institute
Beijing, China
```

---

## MSC 2020 Classification Codes

- **83C57** — Black holes (general relativity)
- **53C21** — Methods of Riemannian geometry (global geometric analysis)
- **35J60** — Nonlinear elliptic equations
- **83C40** — Gravitational energy and conservation laws

---

## Notes for Editors

1. **Paper length:** ~220 pages including appendices. This length is appropriate for a comprehensive treatment of a major open problem.

2. **Appendices:** Eight appendices provide technical foundations that support the main argument but can be treated as supplementary material if needed.

3. **Numerical illustrations:** Appendix A contains Python code for verifying the main inequality on Kerr spacetimes.

---

## Contact

- **Author:** Da Xu
- **Email:** xuda@chinamobile.com
- **Institution:** China Mobile Research Institute, Beijing 100053, China
