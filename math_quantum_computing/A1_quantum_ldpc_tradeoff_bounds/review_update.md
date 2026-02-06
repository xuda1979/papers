# Referee-Style Review: "New Tradeoff Bounds for Quantum LDPC Parameters"

## Summary
The paper studies parameter tradeoffs for geometrically local quantum LDPC stabilizer codes, with an emphasis on the 2D Bravyi–Poulin–Terhal (BPT) bound and how locality constraints preclude simultaneously constant rate and growing distance. The main theorem restates the 2D BPT tradeoff as $d\sqrt{R} \le C\,wL$, and the paper illustrates tightness using the toric code. A short section recalls a standard (classical) expander-to-distance implication to motivate why expander-based quantum LDPC constructions are necessarily non-geometrically-local.

## Strengths
- The exposition is clean and the narrative arc (locality bound \rightarrow toric saturation \rightarrow contrast with good non-local LDPC codes) is easy to follow.
- The toric-code calculation is correct and helps anchor the scaling.
- The remark about the gap between classical Tanner-graph expansion and quantum distance is appropriate and prevents a common confusion.

## Major Issues (need addressing for a research-paper claim)

### 1) Novelty / positioning
As written, the “main result” is a restatement of BPT in 2D rather than a new bound. If the intended contribution is *explicit dependence* on interaction range and stabilizer weight, the paper should:
- Either (a) clearly position itself as an expository note / tutorial, or (b) add a genuinely new statement (e.g., a sharpened dependence, an extension to a different model/assumption, or a new regime).
- Update the title and abstract accordingly if there is no new asymptotic bound beyond known BPT/Bravyi–Terhal-type tradeoffs.

### 2) Locality model and parameter dependence ($L$ and $w$)
Theorem 3.1 depends critically on what “interaction range $L$” means and how it relates to “maximum stabilizer weight $w$”. Right now the dependence $C\,wL$ is asserted as “made explicit in the constant” without a derivation.

Actionable fixes:
- Add a precise definition of geometric locality / range: e.g., qubits embedded in a 2D manifold/lattice with bounded density, and each stabilizer generator supported within a metric ball of diameter $\le L$.
- Explain (even briefly) how BPT’s constant changes under rescaling of interaction range. A standard way is coarse-graining into blocks of linear size $\Theta(L)$ to reduce to a constant-range model; this typically introduces an explicit polynomial dependence on $L$ whose exponent depends on the dimension and the locality model.
- Clarify whether $w$ is an independent input parameter (e.g., bounded-weight checks even if $L$ grows) or whether $w$ implicitly scales like the number of qubits in a radius-$L$ region (often $\Theta(L^2)$ in 2D at bounded density). If the latter, the statement $C\,wL$ should be reconciled with that scaling to avoid double-counting locality.

### 3) Scope mismatch: “tradeoff bounds” vs a single 2D bound
The title and introduction suggest broader “new tradeoff bounds” under combined constraints (bounded weight/degree/locality), but the paper presents one 2D BPT restatement plus a classical expander lemma.

Actionable fixes:
- Either generalize the main theorem to $D>2$ (where Bravyi–Terhal-type bounds give $k d^{2/(D-1)} \le O(n)$ under appropriate locality assumptions) and carefully track how the dependence on range enters, or
- Narrow the scope (title/intro) to “2D locality tradeoff (BPT) with explicit parameter bookkeeping” and remove claims about “advancing the understanding” unless you add new technical content.

### 4) “Expansion forbids low-dimensional embedding” needs a citation or a precise statement
The “Breaking the Barrier” subsection asserts that the interaction graphs of good LDPC constructions have expansion properties “that forbid low-dimensional embeddings (the geometric separator lemma fails).” This is plausible, but currently informal.

Actionable fixes:
- State a concrete separator fact: bounded-degree graphs coming from 2D geometric locality typically admit $O(\sqrt{n})$ separators (or, more generally, $O(n^{1-1/D})$ in $D$ dimensions), whereas expanders have linear-size separators. Then cite a standard separator theorem / expander separator lower bound.
- Be careful with wording: what is forbidden is not “embedding” per se (graphs can be embedded with crossings), but *embedding with bounded edge lengths / geometric locality*.

## Minor / Presentation Issues
- The bibliography mixes `\bibliographystyle{alpha}` with manual `thebibliography`. If staying manual, use labeled `\bibitem[Key]{...}` entries consistently (as already done for BPT/PK in the current `paper.tex`). Alternatively remove `\bibliographystyle{alpha}` to avoid confusion.
- Theorem 3.1 calls the bound “BPT-type locality bound in 2D”; consider explicitly saying it is a restatement of BPT and cite BPT at first mention in the Introduction.
- Theorem 3.2 (expander distance) is correct but currently disconnected from later arguments. Either (a) add a short paragraph explaining exactly how this motivates non-local quantum LDPC constructions, or (b) move it to an appendix / related work.
- Consider adding one sentence clarifying the model class covered by BPT in your usage (stabilizer codes vs general commuting-projector Hamiltonians), and whether you assume bounded local Hilbert space dimension.

## Recommendation
**Weak reject as a “new bounds” research paper**, **accept as a short expository note** *if* the paper is reframed accordingly and the locality model / dependence on $L$ and $w$ is made precise.

## Concrete To-Do List for the Next Revision
1. Define geometric locality and interaction range precisely (one paragraph + a definition).
2. Add a short derivation (or a cited lemma) showing how BPT’s constant depends on interaction range (coarse-graining argument).
3. Reposition novelty: either broaden to $D>2$ with tracked constants, or retitle/rewrite as an exposition.
4. Replace the “separator lemma fails” sentence with a precise separator-vs-expander statement + citation.
