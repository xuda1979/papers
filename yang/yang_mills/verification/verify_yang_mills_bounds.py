#!/usr/bin/env python3
"""
Rigorous Verification of Yang-Mills Mass Gap Bounds
====================================================

This script verifies the key mathematical inequalities used in the proof of the 
Yang-Mills Mass Gap as described in the manuscript (January 2026).

Based on the critical review recommendations, this code focuses on:
1. Turán inequalities for modified Bessel functions
2. Spectral gap bounds via Bessel function ratios
3. Giles-Teper constant verification
4. Strict convexity bounds for the adjoint potential
5. Boundary marginal LSI constant verification

References:
- Appendix R.87: Bessel function bounds
- Appendix R.16: Boundary Marginal Decay
- Appendix R.97: Strict Convexity of Adjoint Potential
- Appendix R.80: Giles-Teper Bound

Author: Verification code for Da Xu's manuscript
Date: January 2026
"""

"""verification/verify_yang_mills_bounds.py

NOTE (Jan 2026):
----------------
This file originally depended on SciPy for Bessel/quad evaluations.
The repository's *default* requirements explicitly avoid SciPy to keep
Windows installs pure-Python-friendly, and SciPy-point-evaluations are not
considered proof-grade enclosures.

For rigorous verification, we delegate to interval-based checkers that use
`mpmath.iv` together with repository-provided certified enclosures.

If you need the old exploratory SciPy numerics, recover them from git history
or create a separate optional script under `verification/extras/`.
"""

import sys

from typing import Tuple, Dict, Any

# Delegate to the strict, proof-oriented verifier(s)
import verify_bounds_monotonic
import verify_rigorous_bounds


# =============================================================================
# SECTION 1: TURÁN INEQUALITIES FOR MODIFIED BESSEL FUNCTIONS
# =============================================================================

def verify_turan_inequality(n_max: int = 5, x_max: float = 20.0) -> Tuple[bool, Dict[str, Any]]:
    """Proof-grade Turán verification (delegates to rigorous interval arithmetic)."""
    passed = verify_rigorous_bounds.verify_turan_inequality(n_max=n_max, x_max=x_max)
    return passed, {"passed": passed, "backend": "mpmath.iv"}


# =============================================================================
# SECTION 2: SU(2) SPECTRAL GAP BOUNDS
# =============================================================================

def verify_su2_gap_bound() -> Tuple[bool, Dict[str, Any]]:
    """Proof-grade SU(2) gap bound verification (delegates)."""
    passed = verify_rigorous_bounds.verify_su2_gap_bound()
    return passed, {"passed": passed, "backend": "mpmath.iv"}


def main() -> int:
    print("=" * 60)
    print("RIGOROUS VERIFICATION: Yang-Mills bounds (delegated)")
    print("=" * 60)
    print("[INFO] Progress: set YM_PROGRESS=1 to show periodic updates (default on).")
    print("[INFO] Progress bar: set YM_TQDM=1 and install tqdm to show a live bar.")

    ok1, _ = verify_turan_inequality()
    ok2, _ = verify_su2_gap_bound()
    ok3 = (verify_bounds_monotonic.main() == 0)

    if ok1 and ok2 and ok3:
        print("[SUCCESS] All delegated rigorous checks passed.")
        return 0

    print("[FAIL] One or more delegated rigorous checks failed.")
    return 1


def run_all_verifications() -> bool:
    """Backwards-compatible API: returns True iff all strict checks pass."""
    return main() == 0


if __name__ == "__main__":
    sys.exit(main())
