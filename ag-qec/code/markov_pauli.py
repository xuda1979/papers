from __future__ import annotations
import numpy as np
from typing import Tuple

def _chain(length: int, p: float, eta: float, rng: np.random.Generator) -> np.ndarray:
    """
    First-order binary Markov chain with marginal p and persistence eta in [0,1).
    P(s_t=1 | s_{t-1}=1) = eta + (1-eta)*p
    P(s_t=1 | s_{t-1}=0) = (1-eta)*p
    """
    s = np.zeros(length, dtype=np.uint8)
    # initialize at stationary dist ~ Bernoulli(p)
    s[0] = 1 if rng.random() < p else 0
    p11 = eta + (1 - eta) * p
    p01 = (1 - eta) * p
    for t in range(1, length):
        if s[t-1] == 1:
            s[t] = 1 if rng.random() < p11 else 0
        else:
            s[t] = 1 if rng.random() < p01 else 0
    return s

def pauli_markov_block(n: int, pX: float, pZ: float, pY: float, eta: float, 
                       rhoZY: float = 0.0, seed: int = 1234) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate X and Z binary error vectors for a length-n CSS block under
    an asymmetric first-order Markov Pauli model.
    By default, X, Z, Y chains are independent; optional rhoZY correlates Z and Y bursts.
    Returns (eX, eZ) in {0,1}^n, after mapping Y into both X and Z flips.
    """
    rng = np.random.default_rng(seed)
    X = _chain(n, pX, eta, rng)
    if rhoZY <= 0:
        Z = _chain(n, pZ, eta, rng)
        Y = _chain(n, pY, eta, rng)
    else:
        # simple coupling: mix Z and Y via a shared latent burst process B
        B = _chain(n, max(pZ, pY), eta, rng)
        Z = np.where(B == 1, (rng.random(n) < (rhoZY + (1-rhoZY)*pZ)).astype(np.uint8),
                     (rng.random(n) < pZ).astype(np.uint8))
        Y = np.where(B == 1, (rng.random(n) < (rhoZY + (1-rhoZY)*pY)).astype(np.uint8),
                     (rng.random(n) < pY).astype(np.uint8))
    # Map Y into both X and Z effective flips
    eX = (X ^ 0) | Y
    eZ = (Z ^ 0) | Y
    return eX.astype(np.uint8), eZ.astype(np.uint8)

def effective_rates(pX: float, pZ: float, pY: float) -> Tuple[float,float]:
    """Effective bit- and phase-flip rates for CSS when Y exists."""
    return pX + pY, pZ + pY
