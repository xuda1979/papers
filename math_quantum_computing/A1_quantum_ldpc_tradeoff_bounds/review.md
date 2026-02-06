# Review of "New Tradeoff Bounds for Quantum LDPC Parameters"

## Summary
The paper presents new theoretical bounds for Quantum LDPC codes, focusing on the tradeoffs between rate, distance, and geometric locality. It attempts to generalize and sharpen limitations for codes embedded in $D$-dimensional lattices using separator theorems and spectral graph theory.

## Major Comments

### 1. Theorem 3.1 and BPT Bound Accuracy
The proof of Theorem 3.1 cites the Bravyi-Poulin-Terhal (BPT) bound as $d \cdot k^{1/D} \leq O(L \cdot n)$. 
However, the standard BPT bound (PRL 2010) for 2D local commuting projector codes is $k d^2 \leq O(n)$.
Comparing the two for $D=2$:
*   **BPT (Standard):** $d\sqrt{k} \leq O(\sqrt{n})$
*   **Paper's Equation:** $d\sqrt{k} \leq O(n)$ (derived from $d (k/n)^{1/2} \leq n^{1/2}$ which implies $d\sqrt{k} \leq n$)

The bound derived in the paper is significantly looser (by a factor of $\sqrt{n}$ in 2D) than the established BPT result. If the intention is to "sharpen" bounds, this derivation needs to be revisited. The discrepancy suggests that the starting premise $d \cdot k^{1/D} \leq O(L \cdot n)$ might be incorrect or applies to a different class of codes (e.g., non-commuting without the BPT constraints).

### 2. Example 4.1: Hypergraph Product Codes and Locality
The paper uses Hypergraph Product (HGP) codes constructed from classical expanders as an example of a "2D" code matching the bound.
*   **Issue:** HGP codes based on expanders are **not** geometrically local in 2D. While they can be visualized on a grid, the check weights or interaction lengths do not stay constant; they inherit the non-locality of the expander graph.
*   **Consequence:** If the code is not local ($L$ is not constant), the bound's dependence on $L$ becomes dominant. Claiming these codes "match" a 2D locality bound is misleading because they violate the premise of fixed $L$ in a 2D metric. They are better described as "non-local" or having an effective dimension that isn't simple Euclidiean 2D.

## Minor Comments

*   **Bibliography**: The file uses manual `\bibitem` entries inside a `thebibliography` environment but also declares `\bibliographystyle{alpha}`. The style command is typically ignored when items are manual. To ensure correct formatting or use of the style, one should either use a `.bib` file with BibTeX or ensure the manual labels match the desired style (e.g., `[BPT10]`).
*   **Citation Consistency**: The citation `\cite{PK}` is used, but BPT is referred to largely by name in the intro. Ensure `\cite{BPT}` is used on first mention.
*   **Introduction flow**: The introduction mentions "sharpen known bounds," but as noted above, the main result may actually be looser. This claim should be verified.

## Recommendations
1.  **Re-verify the BPT derivation** in Proof 3.1. Ensure the scaling with $n$ is correct for the specific code class (commuting vs non-commuting).
2.  **Clarify the Locality of HGP codes** in Example 4.1. Explicitly state that HGP codes of expanders are non-local, or switch the example to Toric codes (HGP of rings) which are truly local but have poor parameters ($k=2, d=\sqrt{n}$).
3.  **Fix the discrepancy** between the claim of "sharpening" and the derived formula.
