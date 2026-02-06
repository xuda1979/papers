"""Phase 2 compatibility shim for interval arithmetic.

IMPORTANT:
This file intentionally re-exports the *canonical* Interval implementation
from `verification/interval_arithmetic.py`.

Historically Phase 2 had its own Interval implementation, which risks
subtle divergence from the directed-rounding rules used elsewhere.
All certificate-grade runs must use a single source of truth.
"""

from interval_arithmetic import Interval  # noqa: F401