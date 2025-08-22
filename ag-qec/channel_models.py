#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Channel and error-model utilities for AG-QEC simulations.

Key utilities:
  - attenuation_linear(alpha_db_per_km, L_km): correct linear attenuation.
  - depolarizing_probs(p): returns [1-p, p/3, p/3, p/3].
  - asymmetric_probs(pX, pY, pZ): returns [1-(pX+pY+pZ), pX, pY, pZ].
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

def attenuation_linear(alpha_db_per_km: float, L_km: float) -> float:
    """
    Convert fiber attenuation in dB/km over a length L_km to a linear
    power ratio A in [0,1]:
        A = 10^(-(alpha_db_per_km * L_km)/10).
    """
    if alpha_db_per_km < 0 or L_km < 0:
        raise ValueError("alpha_db_per_km and L_km must be non-negative.")
    return 10.0 ** (-(alpha_db_per_km * L_km) / 10.0)

def depolarizing_probs(p: float) -> np.ndarray:
    """
    Symmetric depolarizing model:
        P(I)=1-p,  P(X)=P(Y)=P(Z)=p/3
    """
    if p < 0 or p > 1:
        raise ValueError("p must be in [0,1].")
    return np.array([1.0 - p, p/3.0, p/3.0, p/3.0], dtype=float)

def asymmetric_probs(pX: float, pY: float, pZ: float) -> np.ndarray:
    """
    Asymmetric Pauli error model with independent per-qubit categorical draw:
        P(I)=1-(pX+pY+pZ),  P(X)=pX,  P(Y)=pY,  P(Z)=pZ
    """
    for v in (pX, pY, pZ):
        if v < 0 or v > 1:
            raise ValueError("probabilities must be in [0,1].")
    p = pX + pY + pZ
    if p >= 1:
        raise ValueError("pX+pY+pZ must be < 1.")
    return np.array([1.0 - p, pX, pY, pZ], dtype=float)

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (lo, hi).
    """
    if n <= 0:
        return (0.0, 1.0)
    phat = k / float(n)
    denom = 1.0 + (z*z)/n
    center = (phat + (z*z)/(2.0*n)) / denom
    half_width = z * np.sqrt((phat*(1.0 - phat)/n) + (z*z)/(4.0*n*n)) / denom
    lo = max(0.0, center - half_width)
    hi = min(1.0, center + half_width)
    return (lo, hi)

def sample_pauli_iid(
    rng: np.random.Generator, n: int, probs: np.ndarray
) -> np.ndarray:
    """
    Sample IID Pauli outcomes for n qubits.
    Outcomes are integers: 0->I, 1->X, 2->Y, 3->Z.
    """
    if probs.shape != (4,):
        raise ValueError("probs must be length-4 array [P(I),P(X),P(Y),P(Z)].")
    if abs(probs.sum() - 1.0) > 1e-12:
        raise ValueError("probs must sum to 1.")
    return rng.choice(4, size=n, p=probs)
