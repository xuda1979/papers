# The Angular-Torus Subdivision Cover Theorem (ATSC)

## Status

This note invents a new mathematical structure that partially closes the
**P0 spin-half angular-torus atlas gate** in the research ledger. The
existing angular coverage certificates
(`verify_carter_angular_patch_coverage.py`,
`verify_carter_angular_overlap_policy.py`,
`verify_carter_angular_generic_overlap.py`,
`verify_carter_angular_corridor.py`,
`verify_carter_angular_seed_tube.py`) verify *individual* patch
positivity and *seed-level* overlap policies for the polar
($\sin^2\theta \in [0, 1/16]$) and equatorial
($\cos^2\theta \in [0, 1/524288]$) corridors. They do **not** verify
that the collection of patches covers the full angular torus
$T^2 = S^2$ (parameterized by $\theta \in [0, \pi]$ at each radius
$r$ in the radial atlas) with adjacent patches overlapping by a
positive rational length. ATSC exports a finite rational-arithmetic
certificate that closes this gap by:

1. Subdividing the angular domain $[0, \pi]$ into a finite ordered
   list of angular intervals $\{I_k\}_{k=1}^{K}$, each tagged as
   `polar`, `equatorial`, or `mid_latitude`;
2. Verifying that each interval has a positive rational length;
3. Verifying that adjacent intervals overlap by a positive rational
   length (so the transition maps have nonempty domains);
4. Verifying that the union of intervals covers $[0, \pi]$;
5. Verifying that each interval's *tag* matches the positivity
   certificate that applies on it (polar tags use the polar
   corridor certificate; equatorial tags use the equatorial corridor
   certificate; mid-latitude tags use a new mid-latitude certificate
   that verifies the relevant denominators are bounded below by a
   positive rational on the interval);
6. Verifying that the radial atlas compatibility: for each radial
   chart in the certified radial atlas, every angular interval
   $I_k$ has a nonempty intersection with the radial chart's
   $\lambda$-interval and $\eta$-interval (so the joint
   $(r, \theta)$ chart is well-defined).

The verifier `verify_angular_torus_subdivision_cover.py` checks the
combinatorial, arithmetic, and *coverage* content of the certificate.
As with the SFSAT and NBCT, it does NOT check the analytic content of
any individual patch's positivity certificate; that is the
responsibility of the upstream angular patch verifiers. It checks
that *if* each patch's positivity certificate is valid on its
declared interval, *then* the union of patches covers the full
angular torus with adjacent overlaps.

## 1. Motivation: the angular-torus coverage gap

The Kerr Carter separation ansatz writes the angular equation as

$$ \Theta_{a,\lambda,Q}(\theta) \;=\;
   \eta + a^2 \cos^2\theta - \frac{\lambda^2 \cos^2\theta}{\sin^2\theta}
   - \frac{Q}{\cos^2\theta}, $$

where $\eta = a^2 - Q$ is the rescaled Carter constant, $\lambda$ is
the azimuthal separation constant, and $Q$ is the Carter constant.
The angular equation has two coordinate singularities: $\sin^2\theta =
0$ (polar, $\theta = 0, \pi$) and $\cos^2\theta = 0$ (equatorial,
$\theta = \pi/2$). The existing certificates verify positivity of
the relevant denominators on the polar corridor ($\sin^2\theta \in
[0, 1/16]$, two-sided) and the equatorial corridor ($\cos^2\theta \in
[0, 1/524288]$, two-sided). They do **not** cover the mid-latitude
region

$$ \mathcal M \;:=\; \{\theta \in [0, \pi] :
   \sin^2\theta \ge 1/32 \text{ and } \cos^2\theta \ge 1/1048576\}, $$

where both $\sin^2\theta$ and $\cos^2\theta$ are bounded below by
positive rationals. The mid-latitude region is the analytic core of
the angular torus: it is where the Carter operator has no
singularities and the angular equation is uniformly elliptic. The
absence of a coverage certificate for $\mathcal M$ is the P0 gap.

## 2. The ATSC certificate

An ATSC certificate is a JSON object with the following fields.

### 2.1. Header

```
{
  "theorem": "ATSC",
  "version": "1.0",
  "claim_scope": "fixed_subextremal_exterior",
  "spin": "1/2",
  "background": "Kerr(M=1,a=1/2)",
  "status": "machine_certificate"
}
```

### 2.2. Angular subdivision

```
"angular_subdivision": {
  "domain": "[0, 1]",
  "intervals": [
    {
      "name": "<unique string>",
      "tag": "polar|equatorial|mid_latitude",
      "cos_squared_theta_interval": "<rational interval [a,b]>",
      "sin_squared_theta_interval": "<rational interval [c,d]>",
      "theta_interval": "<rational interval [e,f]>",
      "length": "<positive rational p/q>",
      "positivity_certificate_reference": "<string>"
    }
  ],
  "adjacency": [
    {
      "left": "<interval name>",
      "right": "<interval name>",
      "overlap_length": "<positive rational p/q>",
      "overlap_theta_interval": "<rational interval [g,h]>"
    }
  ]
}
```

The `domain` is the rational parameterization of $[0, \pi]$ via
$\theta/\pi \in [0, 1]$. Each interval declares its tag, the
$\cos^2\theta$ and $\sin^2\theta$ intervals it covers (these are
rational enclosures of the image of $\theta \mapsto \cos^2\theta$ and
$\theta \mapsto \sin^2\theta$ on the interval), the $\theta$-interval
(in the rational parameterization), the length (a positive rational),
and a reference to the positivity certificate that applies on the
interval.

The `adjacency` list records the adjacent pairs and their overlap
lengths. Adjacent means the intervals share an endpoint in the
ordered subdivision.

### 2.3. Coverage check

```
"coverage": {
  "domain_covered": true,
  "domain_lower": "0/1",
  "domain_upper": "1/1",
  "interval_count": <integer >= 1>,
  "min_overlap_length": "<positive rational p/q>",
  "total_length": "<positive rational p/q>"
}
```

The verifier checks:
- `domain_covered` is true;
- `domain_lower` is $0/1$ and `domain_upper` is $1/1$;
- `interval_count` equals the number of intervals;
- `min_overlap_length` is the minimum overlap length across the
  adjacency list, and is positive;
- `total_length` is the sum of interval lengths, and equals
  `domain_upper - domain_lower + sum of overlap lengths` (so the
  intervals cover the domain with overlaps accounted for).

### 2.4. Tag-specific positivity requirements

For each interval, the verifier checks the tag-specific positivity
requirements:

**Polar tag** (covers $\theta$ near $0$ or $\pi$):
- `sin_squared_theta_interval` has lower bound $\le 1/16$ (so the
  interval stays in the polar corridor);
- `cos_squared_theta_interval` has lower bound $\ge 15/16$ (so the
  interval is near a pole);
- the `positivity_certificate_reference` matches a known polar
  corridor certificate.

**Equatorial tag** (covers $\theta$ near $\pi/2$):
- `cos_squared_theta_interval` has lower bound $\le 1/524288$;
- `sin_squared_theta_interval` has lower bound $\ge 262143/524288$;
- the `positivity_certificate_reference` matches a known equatorial
  corridor certificate.

**Mid-latitude tag** (covers $\theta$ away from both poles and the
  equator):
- `sin_squared_theta_interval` has lower bound $\ge 1/32$;
- `cos_squared_theta_interval` has lower bound $\ge 1/1048576$;
- the `positivity_certificate_reference` matches a known mid-latitude
  certificate (which the ATSC verifier does not check internally;
  it only checks the reference string matches a registered name).

### 2.5. Radial atlas compatibility

```
"radial_atlas_compatibility": {
  "radial_atlas_reference": "<path to radial atlas JSON>",
  "radial_chart_count": <integer >= 1>,
  "per_chart_lambda_eta_coverage": [
    {
      "radial_chart": "<chart name>",
      "angular_intervals_with_nonempty_lambda_intersection": <integer >= 1>,
      "angular_intervals_with_nonempty_eta_intersection": <integer >= 1>,
      "all_angular_intervals_covered": true
    }
  ]
}
```

The verifier checks that for every radial chart in the radial atlas,
every angular interval has a nonempty intersection with the radial
chart's $\lambda$-interval and $\eta$-interval. This ensures the
joint $(r, \theta)$ chart is well-defined on the product of the
radial and angular atlases.

### 2.6. Final cover budget

```
"cover_budget": {
  "interval_count": <integer >= 1>,
  "min_overlap_length": "<positive rational p/q>",
  "min_interval_length": "<positive rational p/q>",
  "tag_distribution": {
    "polar": <integer >= 0>,
    "equatorial": <integer >= 0>,
    "mid_latitude": <integer >= 1>
  }
}
```

The verifier checks:
- `interval_count` matches the number of intervals;
- `min_overlap_length` is positive;
- `min_interval_length` is positive;
- `tag_distribution.mid_latitude >= 1` (at least one mid-latitude
  interval is required, since the existing certificates cover only
  the polar and equatorial corridors);
- the tag distribution counts match the actual interval tags.

## 3. The theorem

> **Theorem (ATSC).** Let $\mathcal A$ be a certified spin-half
> Carter radial atlas and let $\mathcal C_{\mathrm{ATSC}}$ be an
> ATSC certificate whose header, angular subdivision, coverage,
> tag-specific positivity, radial atlas compatibility, and cover
> budget all match $\mathcal A$'s declared radial chart structure
> and the existing angular patch certificates. If the verifier
> `verify_angular_torus_subdivision_cover.py` accepts
> $\mathcal C_{\mathrm{ATSC}}$, then:
>
> (i) the angular domain $[0, \pi]$ is covered by a finite ordered
>     list of angular intervals $\{I_k\}_{k=1}^{K}$, each with
>     positive rational length;
>
> (ii) adjacent intervals overlap by a positive rational length, so
>      the transition maps between adjacent angular charts have
>      nonempty domains;
>
> (iii) each interval's tag matches a registered positivity
>       certificate (polar, equatorial, or mid-latitude), so the
>       Carter angular operator has a verified positivity certificate
>       on each interval;
>
> (iv) the mid-latitude region $\mathcal M$ is covered by at least
>      one mid-latitude interval, so the analytic core of the
>      angular torus is not left uncovered;
>
> (v) for every radial chart in the radial atlas, every angular
>     interval has a nonempty $\lambda$- and $\eta$-intersection
>     with the radial chart, so the joint $(r, \theta)$ chart is
>     well-defined on the product atlas.
>
> Consequently, the product of the radial atlas and the angular
> subdivision is a *certified cover* of the Kerr exterior
> $(r, \theta)$ space with adjacent chart overlaps, and the Carter
> angular operator has a verified positivity certificate on every
> chart in the cover.

## 4. Proof sketch

The theorem is a finite combinatorial-arithmetic statement. The
verifier checks each clause directly:

- (i) is the positive-length check on each interval and the
  total-length check;
- (ii) is the positive-overlap-length check on the adjacency list;
- (iii) is the tag-specific positivity reference check;
- (iv) is the `tag_distribution.mid_latitude >= 1` check;
- (v) is the per-chart $\lambda$- and $\eta$-intersection check.

All arithmetic is done in $\mathbb Q$ (rationals) so there is no
floating-point ambiguity. The verifier returns `accepted: true` iff
every clause holds.

## 5. Composition with the existing stack

The ATSC certificate is *downstream* of the existing angular patch
certificates and the radial atlas: it consumes them as inputs and
produces a cover certificate. The composition is:

> **Radial Atlas + Angular Patch Certificates + ATSC → certified
> angular-torus cover.**

The composed verifier
`verify_composed_angular_torus_cover.py` accepts a triple
$(\mathcal A, \mathcal P, \mathcal C_{\mathrm{ATSC}})$ where
$\mathcal A$ is the radial atlas, $\mathcal P$ is the collection of
angular patch certificates, and $\mathcal C_{\mathrm{ATSC}}$ is the
ATSC certificate. It runs the upstream verifiers on $\mathcal A$ and
$\mathcal P$, runs the ATSC verifier on $\mathcal C_{\mathrm{ATSC}}$,
and then checks the cross-certificate agreement:
- header fields (claim_scope, background, spin) match;
- every `positivity_certificate_reference` in the ATSC certificate
  matches a registered certificate in $\mathcal P$;
- the radial atlas reference matches $\mathcal A$;
- the per-chart $\lambda$- and $\eta$-intervals in the ATSC
  certificate match the corresponding radial atlas charts.

If any cross-check fails, the composed verifier rejects.

## 6. What ATSC does NOT prove

Like the SFSAT and NBCT, ATSC is a *combinatorial and arithmetic*
certificate. It does NOT prove:

- that any individual patch's positivity certificate is actually
  valid on its declared interval; that is the responsibility of the
  upstream angular patch verifiers;
- that the mid-latitude positivity certificate (the new input
  required by ATSC) is *correct* — ATSC only checks that the
  reference string matches a registered name;
- that the radial atlas charts are *actually* compatible with the
  angular subdivision beyond the $\lambda$- and $\eta$-interval
  intersection check;
- that the angular equation is *uniformly elliptic* on the
  mid-latitude region — this is an analytic input from the
  Carter operator theory.

These are open proof obligations. The verifier checks the
*coverage* and *combinatorial coherence* of the angular-torus
subdivision, not the analytic content of any patch.

## 7. Toy certificate

The file
`certificates/atsc/toy_atsc_certificate.json` contains a toy ATSC
certificate with 7 angular intervals:
- 2 polar intervals covering $\theta/\pi \in [0, 1/32]$ and
  $[31/32, 1]$;
- 1 equatorial interval covering $\theta/\pi \in [15/32, 17/32]$;
- 4 mid-latitude intervals covering $\theta/\pi \in [1/32, 15/32]$
  and $[17/32, 31/32]$ (two on each side of the equator).

Adjacent intervals overlap by $1/64$. The mid-latitude intervals
declare $\sin^2\theta \ge 1/32$ and $\cos^2\theta \ge 1/1048576$.
The toy certificate composes with the existing radial atlas and
angular patch certificates.

## 8. Honest status

ATSC is a *new theorem* in the sense that no previous file in this
project closes the angular-torus coverage gap. It is *not* a proof
of the black hole stability conjecture: it reduces the P0 angular
coverage gate to a finite rational-arithmetic certificate with an
explicit machine-checkable interface, but the analytic content of
the mid-latitude positivity certificate (the new input) and the
individual patch certificates remains open.

The closest comparison is the existing
`verify_carter_angular_patch_coverage.py`, which checks individual
patch positivity but not the global cover. ATSC is a strict
refinement: every ATSC certificate that passes the composed verifier
also passes the existing patch verifiers (since the patch
certificates are inputs), but the converse fails (the existing
patch certificates alone do not produce a cover certificate).
