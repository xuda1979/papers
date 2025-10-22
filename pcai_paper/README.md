# Proof-Carrying AI (PCAI) — Minimal Prototype

This repo contains a small LaTeX paper and a reference Python implementation of a verifier that checks machine-readable certificates.

## Quickstart

```bash
# Demo: universal property check + contract-checked division
python3 pcai_reference_impl.py demo --out text

# Verify sum of first N integers (∀ n∈[0..N], implementation == spec)
python3 pcai_reference_impl.py sum --N 10000 --out json

# Verified division with configurable tolerances and output format
python3 pcai_reference_impl.py divide 7 3 --abs-tol 1e-12 --rel-tol 1e-9 --out digest
```

Each accepted result prints either:
- **text**: A human-readable line with the SHA-256 digest and full certificate JSON,
- **json**: Just the full certificate JSON,
- **digest**: Just the SHA-256 digest of the *canonical* certificate body.

> **Determinism.** The digest is computed from a canonicalized JSON body that **excludes the `verified_at` timestamp**, so identical proofs hash to the same value across runs.

## Standalone certificate verifier

This repo now includes a tiny, dependency-free verifier for PCAI certificates:

```bash
python3 pcai_verify.py path/to/cert.json --show-body
```

What it does:

- Recomputes the **SHA-256 digest** of the canonicalized certificate body.
- Confirms that `cert["digest"]` (if present) matches the recomputed value.
- Optionally verifies an **Ed25519 signature** _if_ `PyNaCl` is available
  (keeps the repository dependency-free; signature check is skipped otherwise).
- Rejects non-finite floats (`NaN`, `Infinity`) to avoid ambiguous encodings.

**Determinism.** Canonicalization uses JSON with **sorted keys** and **no
insignificant whitespace**. The digest is computed over the certificate body
**excluding** volatile fields like `verified_at`, `digest`, and `signature`.
This closely follows the intent of JSON Canonicalization Scheme (RFC 8785),
without adding dependencies.

### Example

```bash
# Verify a certificate file, show the canonical body used for hashing.
python3 pcai_verify.py examples/cert.json --show-body
```

### Programmatic use

```python
from pcai_canonical import canonical_body, canonicalize_json, digest_for_certificate

body = canonical_body(cert)             # drops verified_at, digest, signature
canon = canonicalize_json(body)         # deterministic JSON string
digest = digest_for_certificate(cert)   # sha256 hex
```

### Notes

- You can import these helpers in `pcai_reference_impl.py` with:
  ```python
  from pcai_canonical import canonical_body, canonicalize_json, digest_for_certificate
  ```
  No behavior change is required elsewhere.

## What’s in here

- `pcai_reference_impl.py`: Two certificate styles
  - **forall_range**: checks `fn(n) == spec(n)` for all `n` in `[0..N]` with first counterexample recorded
  - **contract_trace**: checks pre/post-conditions for operations (e.g., division) using `math.isclose` with absolute and relative tolerances
- `pcai_paper.tex`: Short paper describing the framework, algorithms, and guarantees

## Testing

```bash
python -m pip install -U pytest
pytest -q
```

## Notes

The prototype is dependency-free and intentionally small so the verifier can be audited easily. For more realism, integrate an SMT solver or proof assistant and attach proof objects to certificates.
