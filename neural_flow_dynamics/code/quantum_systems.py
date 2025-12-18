"""
Quantum Systems for Neural Flow Dynamics
=========================================

This module implements quantum Hamiltonians and dynamics for benchmarking.
Key systems:
- Transverse-Field Ising Model (TFIM)
- Heisenberg XXZ Chain
- 2D Fermi-Hubbard Model

Author: Neural Flow Dynamics Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
import itertools


# ============================================
# Pauli Matrices
# ============================================

# Pauli matrices (complex)
PAULI_I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
PAULI_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
PAULI_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
PAULI_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

# Raising/lowering operators
PAULI_PLUS = (PAULI_X + 1j * PAULI_Y) / 2
PAULI_MINUS = (PAULI_X - 1j * PAULI_Y) / 2


# ============================================
# Utility Functions
# ============================================

def kron_n(matrices: List[torch.Tensor]) -> torch.Tensor:
    """Compute Kronecker product of multiple matrices."""
    result = matrices[0]
    for m in matrices[1:]:
        result = torch.kron(result, m)
    return result


def identity_except(n: int, i: int, op: torch.Tensor) -> torch.Tensor:
    """
    Create operator acting on site i in n-site system.
    Returns I ⊗ ... ⊗ op ⊗ ... ⊗ I
    """
    ops = [PAULI_I] * n
    ops[i] = op
    return kron_n(ops)


def two_site_operator(n: int, i: int, j: int, op_i: torch.Tensor, op_j: torch.Tensor) -> torch.Tensor:
    """
    Create two-site operator acting on sites i and j.
    Returns I ⊗ ... ⊗ op_i ⊗ ... ⊗ op_j ⊗ ... ⊗ I
    """
    ops = [PAULI_I] * n
    ops[i] = op_i
    ops[j] = op_j
    return kron_n(ops)


# ============================================
# Transverse-Field Ising Model
# ============================================

class TransverseFieldIsing:
    """
    Transverse-Field Ising Model (TFIM) Hamiltonian.
    
    H = -J Σ_<ij> σ^z_i σ^z_j - h Σ_i σ^x_i
    
    Parameters:
        n_sites: Number of lattice sites
        J: Coupling strength
        h: Transverse field strength
        periodic: Use periodic boundary conditions
    """
    
    def __init__(
        self,
        n_sites: int,
        J: float = 1.0,
        h: float = 1.0,
        periodic: bool = True
    ):
        self.n_sites = n_sites
        self.J = J
        self.h = h
        self.periodic = periodic
        self.dim = 2 ** n_sites
        
        # Build Hamiltonian
        self.H = self._build_hamiltonian()
    
    def _build_hamiltonian(self) -> torch.Tensor:
        """Construct the full Hamiltonian matrix."""
        H = torch.zeros(self.dim, self.dim, dtype=torch.complex128)
        
        # ZZ interactions
        n_bonds = self.n_sites if self.periodic else self.n_sites - 1
        for i in range(n_bonds):
            j = (i + 1) % self.n_sites
            H -= self.J * two_site_operator(self.n_sites, i, j, PAULI_Z, PAULI_Z)
        
        # Transverse field
        for i in range(self.n_sites):
            H -= self.h * identity_except(self.n_sites, i, PAULI_X)
        
        return H
    
    def ground_state(self) -> Tuple[float, torch.Tensor]:
        """Find ground state via exact diagonalization."""
        eigenvalues, eigenvectors = torch.linalg.eigh(self.H)
        E0 = eigenvalues[0].real.item()
        psi0 = eigenvectors[:, 0]
        return E0, psi0
    
    def time_evolution(self, psi0: torch.Tensor, t: float) -> torch.Tensor:
        """
        Exact time evolution: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        """
        U = torch.matrix_exp(-1j * self.H * t)
        return U @ psi0
    
    def expectation_value(self, psi: torch.Tensor, op: torch.Tensor) -> complex:
        """Compute ⟨ψ|O|ψ⟩."""
        return (psi.conj() @ op @ psi).item()
    
    def magnetization_z(self, psi: torch.Tensor) -> float:
        """Compute average z-magnetization: (1/N) Σ_i ⟨σ^z_i⟩."""
        mag = 0.0
        for i in range(self.n_sites):
            op = identity_except(self.n_sites, i, PAULI_Z)
            mag += self.expectation_value(psi, op).real
        return mag / self.n_sites


# ============================================
# Heisenberg XXZ Chain
# ============================================

class HeisenbergXXZ:
    """
    Heisenberg XXZ Chain Hamiltonian.
    
    H = J Σ_<ij> (σ^x_i σ^x_j + σ^y_i σ^y_j + Δ σ^z_i σ^z_j)
    
    Parameters:
        n_sites: Number of lattice sites
        J: Coupling strength
        delta: Anisotropy parameter (Δ)
        periodic: Use periodic boundary conditions
    """
    
    def __init__(
        self,
        n_sites: int,
        J: float = 1.0,
        delta: float = 1.0,
        periodic: bool = True
    ):
        self.n_sites = n_sites
        self.J = J
        self.delta = delta
        self.periodic = periodic
        self.dim = 2 ** n_sites
        
        self.H = self._build_hamiltonian()
    
    def _build_hamiltonian(self) -> torch.Tensor:
        """Construct the full Hamiltonian matrix."""
        H = torch.zeros(self.dim, self.dim, dtype=torch.complex128)
        
        n_bonds = self.n_sites if self.periodic else self.n_sites - 1
        for i in range(n_bonds):
            j = (i + 1) % self.n_sites
            
            # XX term
            H += self.J * two_site_operator(self.n_sites, i, j, PAULI_X, PAULI_X)
            # YY term
            H += self.J * two_site_operator(self.n_sites, i, j, PAULI_Y, PAULI_Y)
            # ZZ term (with anisotropy)
            H += self.J * self.delta * two_site_operator(self.n_sites, i, j, PAULI_Z, PAULI_Z)
        
        return H
    
    def ground_state(self) -> Tuple[float, torch.Tensor]:
        """Find ground state via exact diagonalization."""
        eigenvalues, eigenvectors = torch.linalg.eigh(self.H)
        E0 = eigenvalues[0].real.item()
        psi0 = eigenvectors[:, 0]
        return E0, psi0
    
    def time_evolution(self, psi0: torch.Tensor, t: float) -> torch.Tensor:
        """Exact time evolution."""
        U = torch.matrix_exp(-1j * self.H * t)
        return U @ psi0


# ============================================
# 2D Fermi-Hubbard Model (Sparse)
# ============================================

class FermiHubbard2D:
    """
    2D Fermi-Hubbard Model Hamiltonian (using sparse matrices).
    
    H = -t Σ_<ij>,σ (c†_iσ c_jσ + h.c.) + U Σ_i n_i↑ n_i↓
    
    Parameters:
        Lx, Ly: Lattice dimensions
        t: Hopping strength
        U: On-site interaction
        periodic: Use periodic boundary conditions
    
    Note: Uses Jordan-Wigner transformation for fermionic operators.
    """
    
    def __init__(
        self,
        Lx: int,
        Ly: int,
        t: float = 1.0,
        U: float = 4.0,
        periodic: bool = True
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.n_sites = Lx * Ly
        self.n_orbitals = 2 * self.n_sites  # Spin up and down
        self.t = t
        self.U = U
        self.periodic = periodic
        
        # Hilbert space dimension (for small systems)
        self.dim = 2 ** self.n_orbitals
        
        # Build sparse Hamiltonian
        self.H_sparse = self._build_hamiltonian_sparse()
    
    def _site_index(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to site index."""
        return y * self.Lx + x
    
    def _orbital_index(self, site: int, spin: int) -> int:
        """Convert (site, spin) to orbital index."""
        return 2 * site + spin  # spin: 0=up, 1=down
    
    def _neighbors(self, site: int) -> List[int]:
        """Get neighboring sites."""
        x = site % self.Lx
        y = site // self.Lx
        
        neighbors = []
        
        # Right neighbor
        if x < self.Lx - 1 or self.periodic:
            neighbors.append(self._site_index((x + 1) % self.Lx, y))
        
        # Up neighbor
        if y < self.Ly - 1 or self.periodic:
            neighbors.append(self._site_index(x, (y + 1) % self.Ly))
        
        return neighbors
    
    def _build_hamiltonian_sparse(self) -> sparse.csr_matrix:
        """
        Build sparse Hamiltonian using Jordan-Wigner transformation.
        
        c†_j = (Π_{k<j} σ^z_k) σ^+_j
        c_j  = (Π_{k<j} σ^z_k) σ^-_j
        """
        # For small systems, build dense then convert
        # For large systems, build directly in sparse format
        
        if self.dim > 2**16:
            raise ValueError(f"System too large for exact diagonalization: {self.n_orbitals} orbitals")
        
        H = np.zeros((self.dim, self.dim), dtype=np.complex128)
        
        # Hopping terms: -t Σ c†_iσ c_jσ + h.c.
        for site in range(self.n_sites):
            for neighbor in self._neighbors(site):
                for spin in range(2):
                    i = self._orbital_index(site, spin)
                    j = self._orbital_index(neighbor, spin)
                    
                    # Add hopping term
                    self._add_hopping_term(H, i, j, -self.t)
        
        # On-site interaction: U Σ n_i↑ n_i↓
        for site in range(self.n_sites):
            i_up = self._orbital_index(site, 0)
            i_down = self._orbital_index(site, 1)
            
            # n_↑ n_↓ term
            self._add_density_density_term(H, i_up, i_down, self.U)
        
        return sparse.csr_matrix(H)
    
    def _add_hopping_term(self, H: np.ndarray, i: int, j: int, amplitude: complex):
        """Add c†_i c_j + h.c. term to Hamiltonian."""
        # Using Jordan-Wigner: c†_i c_j involves string of σ^z operators
        for state in range(self.dim):
            # Check if orbital j is occupied and i is empty
            if (state >> j) & 1 == 1 and (state >> i) & 1 == 0:
                new_state = state ^ (1 << i) ^ (1 << j)
                
                # Jordan-Wigner sign
                min_idx, max_idx = min(i, j), max(i, j)
                sign = 1
                for k in range(min_idx + 1, max_idx):
                    if (state >> k) & 1:
                        sign *= -1
                
                H[new_state, state] += amplitude * sign
                H[state, new_state] += np.conj(amplitude) * sign
    
    def _add_density_density_term(self, H: np.ndarray, i: int, j: int, amplitude: float):
        """Add n_i n_j term to Hamiltonian (diagonal)."""
        for state in range(self.dim):
            if (state >> i) & 1 == 1 and (state >> j) & 1 == 1:
                H[state, state] += amplitude
    
    def ground_state(self) -> Tuple[float, np.ndarray]:
        """Find ground state using sparse diagonalization."""
        from scipy.sparse.linalg import eigsh
        
        eigenvalues, eigenvectors = eigsh(self.H_sparse, k=1, which='SA')
        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]
        
        return E0, psi0
    
    def time_evolution(self, psi0: np.ndarray, t: float) -> np.ndarray:
        """
        Time evolution using sparse matrix exponential.
        |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        """
        return expm_multiply(-1j * self.H_sparse * t, psi0)
    
    def double_occupancy(self, psi: np.ndarray) -> float:
        """
        Compute average double occupancy: (1/N) Σ_i ⟨n_i↑ n_i↓⟩
        """
        double_occ = 0.0
        
        for site in range(self.n_sites):
            i_up = self._orbital_index(site, 0)
            i_down = self._orbital_index(site, 1)
            
            for state in range(self.dim):
                if (state >> i_up) & 1 == 1 and (state >> i_down) & 1 == 1:
                    double_occ += np.abs(psi[state]) ** 2
        
        return double_occ / self.n_sites


# ============================================
# Quantum State Utilities
# ============================================

def computational_basis_state(n_qubits: int, index: int) -> torch.Tensor:
    """Create computational basis state |index⟩."""
    psi = torch.zeros(2 ** n_qubits, dtype=torch.complex128)
    psi[index] = 1.0
    return psi


def random_state(n_qubits: int, seed: Optional[int] = None) -> torch.Tensor:
    """Create random normalized quantum state."""
    if seed is not None:
        torch.manual_seed(seed)
    
    real = torch.randn(2 ** n_qubits)
    imag = torch.randn(2 ** n_qubits)
    psi = real + 1j * imag
    psi = psi / torch.norm(psi)
    
    return psi


def product_state(single_site_states: List[torch.Tensor]) -> torch.Tensor:
    """
    Create product state from single-site states.
    |ψ⟩ = |ψ_1⟩ ⊗ |ψ_2⟩ ⊗ ... ⊗ |ψ_n⟩
    """
    return kron_n(single_site_states)


def all_up_state(n_qubits: int) -> torch.Tensor:
    """Create |↑↑...↑⟩ state (all zeros in computational basis)."""
    return computational_basis_state(n_qubits, 0)


def neel_state(n_qubits: int) -> torch.Tensor:
    """Create Néel state |↑↓↑↓...⟩."""
    index = sum(2**i for i in range(0, n_qubits, 2))
    return computational_basis_state(n_qubits, index)


def ghz_state(n_qubits: int) -> torch.Tensor:
    """Create GHZ state (|00...0⟩ + |11...1⟩) / √2."""
    psi = torch.zeros(2 ** n_qubits, dtype=torch.complex128)
    psi[0] = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    return psi


# ============================================
# Entanglement Measures
# ============================================

def entanglement_entropy(psi: torch.Tensor, subsystem_qubits: List[int], n_qubits: int) -> float:
    """
    Compute entanglement entropy of a subsystem.
    
    S = -Tr(ρ_A log ρ_A)
    
    where ρ_A = Tr_B(|ψ⟩⟨ψ|) is the reduced density matrix.
    """
    # Reshape state into tensor
    psi_tensor = psi.reshape([2] * n_qubits)
    
    # Compute reduced density matrix
    complement = [i for i in range(n_qubits) if i not in subsystem_qubits]
    
    # Trace out complement
    # This is a simplified implementation - full version would use einsum
    n_A = len(subsystem_qubits)
    n_B = len(complement)
    
    # Reorder axes: subsystem first, then complement
    axes_order = subsystem_qubits + complement
    psi_reordered = psi_tensor.permute(axes_order)
    
    # Reshape to (dim_A, dim_B)
    dim_A = 2 ** n_A
    dim_B = 2 ** n_B
    psi_matrix = psi_reordered.reshape(dim_A, dim_B)
    
    # Reduced density matrix: ρ_A = ψ @ ψ†
    rho_A = psi_matrix @ psi_matrix.conj().T
    
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(rho_A).real
    
    # Remove zero eigenvalues for numerical stability
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    
    # Entropy: S = -Σ λ log λ
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
    
    return entropy.item()


# ============================================
# Test / Demo
# ============================================

if __name__ == "__main__":
    print("Testing Quantum Systems")
    print("=" * 50)
    
    # Test TFIM
    print("\n1. Transverse-Field Ising Model (N=6)")
    tfim = TransverseFieldIsing(n_sites=6, J=1.0, h=0.5)
    E0, psi0 = tfim.ground_state()
    print(f"   Ground state energy: {E0:.6f}")
    print(f"   Magnetization: {tfim.magnetization_z(psi0):.6f}")
    
    # Time evolution
    psi_t = tfim.time_evolution(psi0, t=1.0)
    overlap = torch.abs(torch.vdot(psi0, psi_t)) ** 2
    print(f"   |⟨ψ(0)|ψ(t=1)⟩|² = {overlap.item():.6f}")
    
    # Test Heisenberg
    print("\n2. Heisenberg XXZ Chain (N=6)")
    xxz = HeisenbergXXZ(n_sites=6, J=1.0, delta=1.0)
    E0, psi0 = xxz.ground_state()
    print(f"   Ground state energy: {E0:.6f}")
    
    # Test Hubbard (small system)
    print("\n3. 2D Fermi-Hubbard Model (2x2)")
    hubbard = FermiHubbard2D(Lx=2, Ly=2, t=1.0, U=4.0)
    E0, psi0 = hubbard.ground_state()
    print(f"   Ground state energy: {E0:.6f}")
    print(f"   Double occupancy: {hubbard.double_occupancy(psi0):.6f}")
    
    # Time evolution
    psi_t = hubbard.time_evolution(psi0, t=0.5)
    print(f"   Double occupancy at t=0.5: {hubbard.double_occupancy(psi_t):.6f}")
    
    # Test entanglement
    print("\n4. Entanglement Entropy")
    n_qubits = 6
    ghz = ghz_state(n_qubits)
    S = entanglement_entropy(ghz, [0, 1, 2], n_qubits)
    print(f"   GHZ state entropy (half-system): {S:.6f}")
    print(f"   Expected: {np.log(2):.6f}")
    
    print("\n✓ All quantum systems tested!")
