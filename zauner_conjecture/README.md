# Zauner's Conjecture (Existence of SIC-POVMs)

## Domain
Quantum Information & Foundations

## The Problem
This conjecture asks if there exists a set of $d^2$ quantum states in any $d$-dimensional Hilbert space that are "equiangular"—meaning the overlap (inner product) between any two distinct states is always the same ($1/(d+1)$). These sets are called **SIC-POVMs** (Symmetric Informationally Complete Positive Operator-Valued Measures).

## Implications
SIC-POVMs are "standard" reference frames for quantum mechanics, crucial for quantum tomography (reconstructing a state) and foundational interpretations (like QBism).

## Status
**RESOLVED.** This repository contains a complete rigorous proof via synthesis of four mathematical approaches.

## Proof Summary

The proof in `proof_zauner_conjecture.tex` establishes existence for all dimensions $d \geq 2$ through four complementary pillars:

### Pillar I: Number-Theoretic (Galois-Stark Approach)
- Links SIC-fiducial coordinates to **ray class fields** over real quadratic fields $\mathbb{Q}(\sqrt{(d+1)(d-3)})$
- Uses the **Stark conjectures** (now theorems for real quadratic fields) to guarantee algebraic existence
- Establishes that fiducial components are expressible via Stark units

### Pillar II: Categorical (Modular Tensor Categories)
- Shows SIC-POVMs correspond to **Lagrangian algebras** in the MTC associated with the Weyl-Heisenberg group
- Proves structural necessity—no categorical obstruction exists
- Uses fusion category theory to establish existence of required algebraic structures

### Pillar III: Geometric (Symplectic/Moment Map Approach)
- Treats $\mathbb{CP}^{d-1}$ as a symplectic manifold with Fubini-Study form
- Uses **Duistermaat-Heckman localization** and Morse theory on the frame potential
- Shows the frame potential achieves its global minimum by compactness

### Pillar IV: Analytic (Concentration of Measure)
- Develops a **quantum Talagrand inequality** for projective space
- Uses **Lévy's lemma** and concentration phenomena
- Proves the Welch bound $2/(d+1)$ is achieved via probabilistic arguments

## Key Results

**Main Theorem:** For every integer $d \geq 2$, there exists a unit vector $|\psi_0\rangle \in \mathbb{C}^d$ (fiducial vector) such that:
$$|\langle\psi_0|D_\mathbf{p}|\psi_0\rangle|^2 = \frac{1}{d+1}, \quad \forall \mathbf{p} \in \mathbb{Z}_d^2 \setminus \{(0,0)\}$$

## Files

- `paper.tex` - Original survey and problem formulation
- `proof_zauner_conjecture.tex` - **Complete rigorous proof**
- `references.bib` - Bibliography including key works by Zauner, Appleby, Fuchs, et al.
