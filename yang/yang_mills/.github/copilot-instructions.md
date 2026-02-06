# Copilot Instructions — Yang-Mills Mass Gap Verification

## Project Overview

This is a **Computer-Assisted Proof (CAP)** for the Yang-Mills existence and mass gap problem (SU(3), 4D). It pairs a LaTeX manuscript (`single/yang_mills_mass_gap.tex`) with a Python verification suite (`verification/`). The code performs rigorous interval-arithmetic-based verification of the Renormalization Group (RG) flow contraction across three coupling regimes.

## Architecture

| Layer | Key files | Role |
|-------|-----------|------|
| **Interval core** | `verification/interval_arithmetic.py` | `Interval` class with `math.nextafter` outward rounding — single source of truth for all rigorous numerics |
| **Constants derivation** | `rigorous_constants_derivation.py` | Derives pollution constants & bounding norms ab initio; outputs `rigorous_constants.json` |
| **Phase 2 engine** | `full_verifier_phase2.py` | Iterates RG map on "Tube" checking R(T_k) ⊂ T_{k+1} for β ∈ [0.25, 6.0] |
| **Phase 2 internals** | `verification/phase2/{operator_basis,tube_geometry}/` | Operator basis generator, tube definition, ball covering mesh |
| **Jacobian** | `ab_initio_jacobian.py` | Rigorous remainder bounds for RG map Jacobian across crossover |
| **Proof pillars** | `verify_full_proof.py` | Master 6-pillar verifier (stability, continuum, RP, mass gap, Lorentz, Gribov) |
| **Grand pipeline** | `grand_verification_pipeline.py` | Links UV derivation → RG flow → OS reconstruction → transfer matrix → gap |
| **LaTeX export** | `export_results_to_latex.py` | Generates `single/verification_results.tex` with `\newcommand` macros consumed by the paper |
| **Drift detector** | `drift_check_latex_constants.py` | Flags hard-coded numerics in `.tex` files that should come from generated macros |
| **Provenance** | `provenance.py` | SHA-256 hash chain binding artifacts to derivation source files via `.provenance.json` manifests |
| **Certificate runners** | `certificate_runner.py`, `certificate_runner_v2.py` | Deterministic entrypoints that emit timestamped artifact bundles under `verification/artifacts/` |

## Critical Conventions

### Interval Arithmetic Discipline
All numerical bounds **must** use the `Interval` class from `verification/interval_arithmetic.py`. Never use bare floats for rigorous bounds. The class enforces outward rounding (`math.nextafter` toward ±∞) on every operation. Example:
```python
from interval_arithmetic import Interval
beta = Interval(0.40, 0.40)  # point interval
result = beta.sqrt()          # outward-rounded
```

### Provenance System
Every JSON artifact (e.g., `rigorous_constants.json`) must have a companion `.provenance.json` manifest created by `provenance.record_derivation()`. Tests enforce this — especially in **Clay mode** (`proof_status.json` → `"clay_standard": true`), where missing provenance is a hard failure.

### Clay Mode vs. Non-Clay Mode
- Controlled by `verification/proof_status.json` field `"clay_standard"` and env var `YM_STRICT=1`.
- **Clay mode**: provenance is mandatory, theorem-boundary certificates are rejected, all gates must pass.
- **Non-Clay mode**: provenance is advisory, warnings replace failures.
- Tests in `test_clay_mode_enforcement.py` and `test_provenance.py` gate this behavior.

### LaTeX ↔ Code Binding
Verification results flow **one-way**: code → `verification_results.tex` → paper. Never hand-edit numerical constants in `.tex` files. The paper uses macros like `\VerDobrushinNorm`, `\VerBetaStrongMax`, `\VerClayCertified`. Run `drift_check_latex_constants.py` to detect violations.

## Developer Workflows

### Run tests
```powershell
cd verification
pytest                           # all tests
pytest test_certificates.py      # regression: LaTeX macro consistency
pytest test_provenance.py        # provenance chain integrity
pytest test_clay_mode_enforcement.py  # Clay-mode gating
```

### Regenerate verification artifacts
```powershell
cd verification
python rigorous_constants_derivation.py   # step 1: derive constants
python full_verifier_phase2.py            # step 2: Phase 2 tube verification
python export_results_to_latex.py         # step 3: export to LaTeX macros
python record_provenance_manifests.py     # step 4: bind provenance hashes
python certificate_runner_v2.py           # full pipeline + artifact bundle
```

### Dependencies
Pure-Python stack: `numpy`, `mpmath`, `sympy`, `pytest`. No compiled extensions required (see `verification/requirements.txt`). SciPy/Matplotlib are intentionally excluded for portability.

## File Naming Patterns
- `verify_*.py` — standalone verifiers for specific proof pillars
- `test_*.py` — pytest test files (regression gates + unit tests)
- `certificate_*.json` — proof certificates with companion `.provenance.json`
- `*_evidence.py` / `*_evidence.json` — evidence generators and their outputs
- `*_obligations.py` / `*_hypotheses.py` — formal obligation/hypothesis registries

## Things to Avoid
- **Do not** introduce `scipy` or native-compiled dependencies into the core verification path.
- **Do not** hard-code numeric constants in `.tex` files — always use exported macros.
- **Do not** bypass `Interval` arithmetic with bare `float` computation for any bound that enters a certificate.
- **Do not** modify `proof_status.json` without understanding that `"clay_standard": true` triggers strict provenance enforcement across the entire pipeline.
- **Do not** create new JSON artifacts without recording provenance via `provenance.record_derivation()`.
