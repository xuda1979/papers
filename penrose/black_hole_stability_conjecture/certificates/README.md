# Kerr Certificate Scaffold

This directory turns the certificate roadmap in `black_hole_stability_conjecture.tex`
into a concrete verification target.

The verifier is intentionally conservative. It does not infer missing mathematics.
It checks that a proposed certificate contains explicit rational margins for:

- coercive phase-space energy positivity,
- nonsuperradiant Morawetz positivity and absorbable remainders,
- trapped-superradiant Fourier average positivity,
- nonlinear weak-null and modulation estimates,
- compatibility of Sobolev indices, weights, symbol classes, and frequency decomposition.

Run:

```sh
python3 ../verify_kerr_certificate.py examples/toy_certificate.json
```

Run the negative-margin check and unit tests from the repository root:

```sh
python3 verify_kerr_certificate.py certificates/examples/bad_negative_margin.json
python3 -m unittest tests/test_verify_kerr_certificate.py
```

The schema is recorded in `schema/kerr_certificate.schema.json`. The Python verifier
adds semantic checks that JSON Schema cannot express, such as strict domination of
remainder intervals by positivity intervals.

Track current external candidate inputs:

```sh
python3 ../route_status.py candidate_inputs/current_literature.json
```

Generate a toy trapped-torus Fourier margin block:

```sh
python3 ../generate_fourier_certificate.py fourier_inputs/toy_torus_coefficients.json
```

The included `toy_certificate.json` is a sanity check for the verifier format. It is
not a Kerr proof certificate.

To pursue the conjecture route, replace the toy data with Kerr-derived rational
interval bounds and SOS/Gram decompositions in the same schema.
