# FMIC Submission Checklist

Based on official public information from `Frontiers of Mathematics in China` and Higher Education Press policy.

## Verified locally on 2026-04-07

- `pdflatex -interaction=nonstopmode -halt-on-error pi.tex` completed successfully.
- Final manuscript artifact: `pi.pdf` (`43` pages).
- No undefined-reference, overfull-box, underfull-box, or LaTeX error warnings were found in `pi.log`.
- Current root files `pi.tex` and `pi.pdf` match the staged submission snapshot exactly.

## Official sources checked

- Journal home: https://journal.hep.com.cn/fmc
- Aims and scope: https://journal.hep.com.cn/fmc/EN/aims
- HEP editorial policy: https://journal.hep.com.cn/hep/EN/policy
- Online submission: https://mc.manuscriptcentral.com/fmc

## Scope fit

- Topic falls within the journal scope:
  - probability theory
  - combinatorics and graph theory
  - functional analysis / analysis
- Article type should be `Research Article`.

## Manuscript file

- Main TeX source: `pi.tex`
- Compiled PDF: `pi.pdf`
- Current front matter already includes:
  - title
  - author
  - affiliation
  - email
  - abstract
  - keywords
  - MSC classifications
- This matches the front-matter pattern visible in published FMIC articles on the journal site:
  - title
  - author names
  - affiliation
  - abstract
  - keywords
  - MSC
  - corresponding-author email in multi-author papers

## Author declarations required by HEP policy

- Manuscript is original.
- Manuscript is not under consideration elsewhere.
- All authors have read and approved the submission.
- Authorship is final before acceptance.
- Conflicts of interest must be disclosed.

## Materials prepared

- Cover letter draft: `COVER_LETTER_FMIC.txt`
- Manuscript PDF: `pi.pdf`

## Submission metadata to prepare in ScholarOne

- Title:
  - `Critical Dirichlet Spectral Zeta Asymptotics on Graphs of Quadratic Volume Growth`
- Short title:
  - `Critical Dirichlet Spectral Zeta Asymptotics`
- Article type:
  - `Research Article`
- Author name:
  - `Da Xu`
- Affiliation:
  - `China Mobile Research Institute, Beijing, P.R. China`
- Corresponding author email:
  - `xudayj@chinamobile.com`
- Abstract:
  - use the current single-paragraph abstract from `pi.tex`
- Keywords:
  - `spectral zeta`
  - `graph Laplacian`
  - `heat kernel`
  - `quadratic volume growth`
  - `random walk homogenisation`
- MSC 2020:
  - `58J50`
  - `31C20`
  - `60J10`
  - `05C50`
  - `39A12`
  - `05C81`

## Recommended checks before upload

- PDF opens correctly and all pages are present.
- References, theorem numbers, and hyperlinks are stable after final compile.
- No overfull or undefined-reference warnings remain in the final log.
- Title in ScholarOne matches title in PDF exactly.
- Author names and affiliation formatting in ScholarOne match the manuscript.
- Cover letter states originality and no simultaneous submission.
- Conflict-of-interest response in ScholarOne is filled explicitly.

## Remaining manual actions in ScholarOne

- Paste/upload the exact title, abstract, keywords, and MSC entries from `SCHOLARONE_FMIC_METADATA.txt`.
- Confirm the corresponding-author affiliation formatting exactly as intended by the journal form.
- Answer originality and conflict-of-interest declarations explicitly in the submission system.
- Add suggested reviewers only if the form requires them and there is no conflict.
- Do one final visual skim of `pi.pdf` before clicking submit.

## Items not confirmed publicly on the journal site

- No public detailed author-format page with strict manuscript template rules was available from the journal site at the time of checking.
- No public evidence was found that a separate conflict-of-interest paragraph must appear inside the manuscript PDF for this journal.
- Therefore, conflict-of-interest and originality statements should be entered carefully in the submission system and cover letter.
- Because the public instructions are sparse, if ScholarOne requests any extra structured fields beyond the prepared metadata file, those should be completed directly in the form rather than added ad hoc to the manuscript PDF.
