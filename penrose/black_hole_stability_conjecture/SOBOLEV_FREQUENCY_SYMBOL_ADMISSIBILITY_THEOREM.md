# The Sobolev-Frequency-Symbol Admissibility Theorem (SFSAT)

## Status

This note invents a new mathematical structure that closes the **P2
completion-package compatibility gate** in the research ledger. The
existing `verify_unconditional_completion_package.py` checker only
verifies that every proof block declares the *same string* for gauge,
foliation, Sobolev index, weight, frequency decomposition, and symbol
class. It does **not** verify that the shared normalization is
*mathematically coherent*: Sobolev indices large enough to absorb
derivative losses through every intermediate composition stage;
frequency decompositions with enough dyadic layers to support the
declared symbol class; margin dominance uniform across blocks; and
associative composition order. SFSAT exports a finite rational
certificate that closes all five loopholes simultaneously.

The verifier `verify_sobolev_frequency_symbol_admissibility.py`
checks the combinatorial, arithmetic, and *internal-coherence*
content of the certificate. As with the NBCT, it does NOT check the
analytic content of any individual block (which is the responsibility
of the upstream theorems). It checks that *if* each block is proved
with the declared local hypotheses, *then* the composition is
admissible: the shared Sobolev hierarchy is not exhausted at any
intermediate stage, the frequency decomposition supports the symbol
class, and the product of blockwise margin remainders stays above a
positive rational floor.

## 1. Motivation: the five compatibility loopholes

Let the proof stack be the ordered list of blocks
$$ \mathcal B_1 \to \mathcal B_2 \to \cdots \to \mathcal B_K $$
where each block $\mathcal B_k$ is one of the seven required blocks
in the unconditional completion package:
`linear_to_nonlinear_estimates`, `trapped_superradiant_morawetz`,
`curvature_to_metric_reconstruction`, `global_gauge_modulation`,
`nonlinear_weak_null_hierarchy`, `near_extremal_parameter_control`,
`compatibility_assembly`. Each block consumes a Sobolev regularity
budget $d_k \in \mathbb Z_{\ge 0}$ (the `derivative_loss`) and
produces a margin-remainder pair $(m_k, r_k)$ of rational intervals.

The existing checker verifies:
(C0) every block declares the same `normalization` strings;
(C1) $m_k$ is positive and $r_k$ is nonnegative;
(C2) $m_k \succ r_k$ (strict margin dominance, blockwise);
(C3) $\sum_k d_k \le D_{\max}$ (total derivative loss fits in the buffer);
(C4) `available_derivative_buffer` $\ge \sum_k d_k$.

It does **not** verify:

(L1) **Sobolev headroom at every intermediate stage.** If
   $N$ is the shared Sobolev index and the blocks are composed in
   the order $\mathcal B_1 \to \cdots \to \mathcal B_K$, the
   intermediate state after $\mathcal B_k$ must still carry
   $N - \sum_{j \le k} d_j$ derivatives of regularity, and this
   must be $\ge N_{\min}$ for some universal $N_{\min}$ depending
   only on the gauge and symbol class. The total-loss check (C3)
   only ensures $N - \sum_k d_k \ge 0$ at the *end*; intermediate
   stages can still go negative if the order is wrong.

(L2) **Frequency dyadic layer count.** The symbol class
   $S^0_{\mathrm{CW}}$ is defined with respect to a dyadic
   Littlewood-Paley decomposition; the verifier must check that
   the declared `frequency_decomposition` has at least
   $L_{\min}(S^0_{\mathrm{CW}})$ layers with explicit rational
   layer bounds. A bare string match is too weak.

(L3) **Margin-remainder uniform floor.** Blockwise strict
   dominance $m_k \succ r_k$ does not imply a uniform floor
   $\inf_k (m_k - r_k) \ge \mu_{\mathrm{floor}} > 0$. Without
   such a floor the *product* of blockwise slack factors can
   collapse to zero in the limit $K \to \infty$ (or even at
   finite $K$ if the infimum is zero).

(L4) **Associative composition order.** The derivative losses
   $d_k$ are nonnegative integers, so $\sum_k d_k$ is
   order-independent; but the *margin propagation* is not.
   The composition
   $$ \mathcal M_K \;=\; m_K \cdot (m_{K-1} - r_{K-1}) \cdots (m_1 - r_1) $$
   depends on the order in which margins are subtracted from the
   seed budget. The certificate must declare a composition order
   and verify that the margin product stays positive at every
   prefix.

(L5) **Symbol-class regularity scale.** The Sobolev index $N$
   must be at least the symbol-class regularity scale
   $N_{\mathrm{sym}}(S^0_{\mathrm{CW}})$ plus the maximum
   per-block derivative loss. Otherwise the block's symbol-class
   hypothesis is incompatible with the available regularity.

## 2. The SFSAT certificate

A SFSAT certificate is a JSON object with the following fields.

### 2.1. Header

```
{
  "theorem": "SFSAT",
  "version": "1.0",
  "claim_scope": "<must match the completion package claim_scope>",
  "background": "<must match the completion package normalization.background>",
  "spin": "<must match>",
  "status": "machine_certificate"
}
```

### 2.2. Sobolev hierarchy

```
"sobolev_hierarchy": {
  "N_shared": <integer >= 1>,
  "N_min": <integer >= 1>,
  "N_symbol_class": <integer >= 1>,
  "max_per_block_derivative_loss": <integer >= 0>,
  "headroom_floor": <nonnegative rational string "p/q">
}
```

The certificate must verify:
- $N_{\mathrm{shared}} \ge N_{\mathrm{symbol\_class}} + \max_k d_k$;
- $N_{\mathrm{shared}} - \sum_{j \le k} d_j \ge N_{\min}$ for every
  prefix $k = 1, \ldots, K$;
- $N_{\min} \ge N_{\mathrm{symbol\_class}} + \max_k d_k -
  \mathrm{headroom\_floor}$.

The last condition prevents the certificate from declaring a tiny
$N_{\min}$ that effectively vacates the headroom requirement.

### 2.3. Frequency decomposition

```
"frequency_decomposition": {
  "name": "<must match completion package normalization.frequency_decomposition>",
  "layer_count": <integer >= 1>,
  "layer_count_minimum_for_symbol_class": <integer >= 1>,
  "layer_bounds": [
    {"layer_index": <integer >= 0>, "lower": "<rational>", "upper": "<rational>"}
  ],
  "covers_full_range": true
}
```

The verifier checks:
- $\text{layer\_count} \ge \text{layer\_count\_minimum\_for\_symbol\_class}$;
- each layer bound is a positive rational interval $[a, b]$ with
  $a \le b$;
- adjacent layers either touch or overlap with positive overlap;
- the union of layer bounds covers $[1, \infty)$ in the sense that
  the lower bound of layer 0 is $\le 1$ and the upper bound of the
  last layer is $\ge$ a declared rational terminal frequency
  $\lambda_{\max}$;
- `covers_full_range` is true.

### 2.4. Symbol class

```
"symbol_class": {
  "name": "<must match normalization.symbol_class>",
  "order": <integer, must be 0 for S^0>,
  "regularity_scale": <integer >= 1>,
  "decay_rate": <nonnegative rational string>,
  "rational_symbol_bound": "<positive rational p/q>"
}
```

The verifier checks:
- `order` is 0 (the CWCC symbol class is order zero);
- `regularity_scale` matches `sobolev_hierarchy.N_symbol_class`;
- `decay_rate` is a nonnegative rational;
- `rational_symbol_bound` is a positive rational.

### 2.5. Blockwise margin schedule

```
"block_margins": [
  {
    "name": "<one of the seven required block names>",
    "derivative_loss": <integer >= 0>,
    "positive_margin": "<rational interval [a,b]>",
    "remainder_bound": "<rational interval [c,d]>",
    "margin_floor": "<positive rational p/q>",
    "composition_position": <integer in 1..K>
  }
]
```

The verifier checks:
- the seven entries are exactly the seven required block names;
- the `derivative_loss` matches the corresponding block in the
  completion package;
- `positive_margin` and `remainder_bound` match the corresponding
  block;
- $\inf (m_k - r_k) \ge \text{margin\_floor}$, computed as a rational
  interval difference;
- the `composition_position` values are a permutation of
  $\{1, \ldots, K\}$.

### 2.6. Composition order

```
"composition_order": [
  <block name 1>, <block name 2>, ..., <block name K>
]
```

This is the order in which margins are composed. The verifier
checks:
- `composition_order` is a permutation of the seven block names;
- the product
  $$ \mathcal M_{\mathrm{prefix}}(k)
     \;=\; \prod_{j=1}^{k}
       (m_{\sigma(j)} - r_{\sigma(j)}) $$
  is positive at every prefix $k = 1, \ldots, K$, where $\sigma$
  is the composition order.

### 2.7. Final admissibility budget

```
"admissibility_budget": {
  "margin_floor": "<positive rational p/q>",
  "headroom_floor": "<nonnegative rational p/q>",
  "product_lower_bound": "<positive rational p/q>",
  "total_derivative_loss": <integer >= 0>,
  "max_total_derivative_loss": <integer >= 0>,
  "available_derivative_buffer": <integer >= 0>
}
```

The verifier checks:
- `margin_floor` matches the infimum of the blockwise `margin_floor`
  values;
- `headroom_floor` matches `sobolev_hierarchy.headroom_floor`;
- `product_lower_bound` is a positive rational lower bound on
  $\mathcal M_{\mathrm{prefix}}(K)$, the full composition product;
- `total_derivative_loss` equals the sum of blockwise derivative
  losses;
- `total_derivative_loss <= max_total_derivative_loss`;
- `available_derivative_buffer >= total_derivative_loss`.

## 3. The theorem

> **Theorem (SFSAT).** Let $\mathcal P$ be an unconditional
> Kerr-stability completion package and let $\mathcal C_{\mathrm{SFSAT}}$
> be a SFSAT certificate whose header, Sobolev hierarchy, frequency
> decomposition, symbol class, blockwise margin schedule, and
> composition order all match $\mathcal P$'s declared normalization
> and proof blocks. If the verifier
> `verify_sobolev_frequency_symbol_admissibility.py` accepts
> $\mathcal C_{\mathrm{SFSAT}}$, then:
>
> (i) the shared Sobolev index $N$ is at least the symbol-class
>     regularity scale plus the maximum per-block derivative loss,
>     so every block's symbol-class hypothesis is compatible with
>     the available regularity;
>
> (ii) for every prefix $k = 1, \ldots, K$ of the composition order,
>      the intermediate Sobolev regularity
>      $N - \sum_{j \le k} d_{\sigma(j)}$ is at least $N_{\min} > 0$,
>      so the intermediate state never drops below the minimum
>      regularity required by the gauge and symbol class;
>
> (iii) the frequency decomposition has at least
>       $L_{\min}(S^0_{\mathrm{CW}})$ dyadic layers with positive
>       adjacent overlap, so the symbol class is well-defined on
>       the declared decomposition;
>
> (iv) for every prefix $k = 1, \ldots, K$, the margin product
>      $\mathcal M_{\mathrm{prefix}}(k)$ is bounded below by
>      $\mu_{\mathrm{floor}}^k > 0$, so the composed slack does
>      not collapse;
>
> (v) the total derivative loss fits in the available buffer, and
>     the final composition product is bounded below by a positive
>     rational.
>
> Consequently, the composition of the seven blocks in the declared
> order is *admissible* in the sense that the shared Sobolev
> hierarchy, frequency decomposition, and symbol class are
> internally coherent across the entire stack.

## 4. Proof sketch

The theorem is a finite combinatorial-arithmetic statement. The
verifier checks each clause directly:

- (i) is the inequality $N_{\mathrm{shared}} \ge N_{\mathrm{symbol\_class}} + \max_k d_k$;
- (ii) is the family of inequalities
  $N_{\mathrm{shared}} - \sum_{j \le k} d_{\sigma(j)} \ge N_{\min}$
  for $k = 1, \ldots, K$, computed by exact rational arithmetic;
- (iii) is the layer-count check and the adjacent-overlap check
  on the layer bounds;
- (iv) is the prefix-product check, computed by exact rational
  interval arithmetic;
- (v) is the total-loss and buffer check.

All arithmetic is done in $\mathbb Q$ (rationals) so there is no
floating-point ambiguity. The verifier returns `accepted: true`
iff every clause holds.

## 5. Composition with the unconditional completion package

The SFSAT certificate is *downstream* of the unconditional
completion package: it consumes the package's `normalization` and
`proof_blocks` as inputs and produces an admissibility certificate.
The composition is:

> **Unconditional Completion Package + SFSAT → admissible
> compatibility.**

The composed verifier
`verify_composed_completion_with_sfsat.py` accepts a pair
$(\mathcal P, \mathcal C_{\mathrm{SFSAT}})$, runs
`verify_unconditional_completion_package.py` on $\mathcal P$,
runs `verify_sobolev_frequency_symbol_admissibility.py` on
$\mathcal C_{\mathrm{SFSAT}}$, and then checks the cross-certificate
agreement:
- header fields (claim_scope, background, spin) match;
- the Sobolev index in `sobolev_hierarchy.N_shared` matches
  `normalization.sobolev_index`;
- each entry in `block_margins` matches the corresponding
  `proof_blocks` entry (derivative_loss, positive_margin,
  remainder_bound);
- `composition_order` is a permutation of the seven block names
  appearing in `proof_blocks`;
- `admissibility_budget.total_derivative_loss` matches the
  total derivative loss computed from `proof_blocks`.

If any cross-check fails, the composed verifier rejects.

## 6. What SFSAT does NOT prove

Like the NBCT, SFSAT is a *combinatorial and arithmetic*
certificate. It does NOT prove:

- that any individual block $\mathcal B_k$ is actually proved;
  that is the responsibility of the upstream theorem/certificate
  (CWCC, CWQP-AB, RRBB, NBCT, etc.);
- that the declared `N_symbol_class` is the *correct* regularity
  scale for the Kerr Carter-Wick symbol class — this is an
  analytic input from the CWCC theorem;
- that the declared `layer_count_minimum_for_symbol_class` is
  the *correct* minimum — this is an analytic input from the
  frequency-decomposition theorem;
- that the declared `composition_order` is the *canonical* order
  in which the blocks must be composed — only that the declared
  order is admissible.

These are open proof obligations. The verifier checks the
*internal coherence* of the compatibility certificate, not the
analytic content of any block.

## 7. Toy certificate

The file
`certificates/sfsat/toy_sfsat_certificate.json` contains a toy
SFSAT certificate compatible with
`certificates/completion_packages/toy_unconditional_completion_package.json`.
The toy certificate declares:
- $N_{\mathrm{shared}} = 16$, $N_{\min} = 8$,
  $N_{\mathrm{symbol\_class}} = 4$, $\max_k d_k = 2$,
  headroom floor $1/1$;
- frequency decomposition with 8 dyadic layers (minimum for the
  symbol class: 4);
- symbol class $S^0_{\mathrm{CW}}$ with regularity scale 4,
  decay rate $1/1$, rational symbol bound $1/1$;
- blockwise margin floor $1/32$ for every block;
- composition order:
  `near_extremal_parameter_control` →
  `trapped_superradiant_morawetz` →
  `linear_to_nonlinear_estimates` →
  `curvature_to_metric_reconstruction` →
  `global_gauge_modulation` →
  `nonlinear_weak_null_hierarchy` →
  `compatibility_assembly`;
- admissibility budget: margin floor $1/32$, headroom floor $1/1$,
  product lower bound $(1/32)^7 = 1/2^{35}$, total derivative loss
  6, max total derivative loss 8, available derivative buffer 9.

The verifier accepts the toy certificate and the composed verifier
accepts the pair (toy package, toy SFSAT certificate).

## 8. Honest status

SFSAT is a *new theorem* in the sense that no previous file in
this project closes the five compatibility loopholes (L1)-(L5)
simultaneously. It is *not* a proof of the black hole stability
conjecture: it reduces the P2 compatibility gate to a finite
rational-arithmetic certificate with an explicit machine-checkable
interface, but the analytic content of each block remains open.

The closest comparison is the existing
`verify_unconditional_completion_package.py`, which checks (C0)-(C4)
but not (L1)-(L5). SFSAT is a strict refinement: every SFSAT
certificate that passes the composed verifier also passes the
existing completion-package verifier, but the converse fails
(the toy package alone passes the existing verifier but only
passes the composed verifier after attaching a SFSAT certificate).
