"""
Safety utilities: action masking, confidence-based fallback (stubs).
In this skeleton, safety is handled inside the Gym wrapper; this file provides hooks.
"""
from __future__ import annotations

def low_confidence(prob: float, threshold: float=0.2) -> bool:
    return prob < threshold
