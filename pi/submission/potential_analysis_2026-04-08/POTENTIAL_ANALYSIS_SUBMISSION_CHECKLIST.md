# Potential Analysis Submission Checklist

Based on the official Springer journal page and submission guidelines for `Potential Analysis`.

## Verified locally on 2026-04-08

- `pdflatex -interaction=nonstopmode -halt-on-error pi.tex` completed successfully.
- Final manuscript artifact: `pi.pdf` (`43` pages).
- The manuscript includes a `Statements and Declarations` section covering:
  - funding
  - competing interests
  - data availability
- The abstract is within the Springer guideline range (`204` words).
- The manuscript provides `5` keywords.
- The manuscript front matter includes title, author, affiliation, email, abstract, keywords, and MSC classifications.
- Current root files `pi.tex` and `pi.pdf` are the versions intended for submission.

## Official sources checked

- Journal page: https://link.springer.com/journal/11118
- Submission guidelines: https://link.springer.com/journal/11118/submission-guidelines
- Aims and scope / journal overview: https://link.springer.com/journal/11118

## Scope fit

- The journal scope explicitly covers the interaction of:
  - potential theory
  - probability theory
  - geometry
  - functional analysis
- The paper fits this profile through:
  - Dirichlet Laplacians and spectral zeta asymptotics
  - heat-kernel and Green-function analysis on graphs
  - probabilistic/random-walk input
  - discrete geometric hypotheses and consequences

## Springer requirements checked against the manuscript

- Abstract length guideline:
  - satisfied (`204` words)
- Keywords guideline:
  - satisfied (`5` keywords)
- MSC classifications:
  - present in the manuscript
- Statements and declarations:
  - added to the manuscript
- PDF + source submission:
  - `pi.pdf` and `pi.tex` are prepared

## Materials prepared

- Manuscript PDF: `pi.pdf`
- Main TeX source: `pi.tex`
- Cover letter: `COVER_LETTER_POTENTIAL_ANALYSIS.txt`
- Submission metadata sheet: `POTENTIAL_ANALYSIS_SUBMISSION_METADATA.txt`
- Optional reviewer suggestions: `POTENTIAL_ANALYSIS_SUGGESTED_REVIEWERS.txt`
- Package readme: `POTENTIAL_ANALYSIS_SUBMISSION_README.txt`

## Recommended submission choices

- Article type:
  - `Original Paper`
- Corresponding author:
  - `Da Xu`
- Journal fit description:
  - emphasize the interaction of potential theory, probability on graphs, heat-kernel analysis, and Dirichlet spectral asymptotics

## Remaining manual actions in the Springer submission system

- Open the current `Potential Analysis` submission portal from the journal page.
- Choose the article type closest to `Original Paper`.
- Paste the title, abstract, keywords, MSC, and declaration text from `POTENTIAL_ANALYSIS_SUBMISSION_METADATA.txt`.
- Upload `pi.pdf`.
- Upload `pi.tex` and any requested source files if the system asks for source submission at the initial stage.
- Upload or paste `COVER_LETTER_POTENTIAL_ANALYSIS.txt` if requested.
- Enter author-contribution text if the portal asks for it explicitly.
- Add reviewer suggestions only if requested and only if conflict-free.

## Notes / risks

- The manuscript uses `amsart` rather than a Springer template. The current public guidelines do not make the Springer template a hard prerequisite for initial submission, but the journal may request template conversion later in the review or production process.
- No nonstandard local style file is required by `pi.tex` for the current build.
- The old `ems-journal.sty` file in the repo is not used by this manuscript.
