# Math-Forward Quantum Computing Papers

This folder contains paper projects organized by mathematical domain. Each subfolder represents a concrete, theorem-driven paper idea.

## Organization

**Formula:** (Math Toolbox) × (Quantum Subfield)

---

## 1) Quantum Error Correction: Coding Theory, Combinatorics, Topology

- `A1_quantum_ldpc_tradeoff_bounds` — New tradeoff bounds for quantum LDPC parameters
- `A2_homological_product_constructions` — Homological-product constructions with explicit asymptotics
- `A3_decoding_guarantees_ldpc` — Decoding guarantees for code families (provable)
- `A4_hardware_aware_code_constraints` — Hardware-aware code constraints as a math problem

## 2) Tensor Networks, Graph Theory, Circuit Simulation

- `B1_treewidth_circuit_complexity` — Complexity bounds via treewidth/pathwidth of circuit graphs
- `B2_random_circuit_moments` — Exact/symbolic moment calculations for random circuits
- `B3_tensor_network_truncation_bounds` — Rigorous error bounds for tensor-network truncation

## 3) Hamiltonian Complexity and Operator Theory

- `C1_local_hamiltonian_complexity` — Restricted Local Hamiltonian variants: sharp complexity frontiers
- `C2_lindbladian_spectral_gap` — Spectral gap bounds and mixing for local Lindbladians
- `C3_commuting_hamiltonians_algebra` — Commuting Hamiltonians and algebraic structure

## 4) Complexity Theory and Lower Bounds

- `D1_quantum_query_complexity` — Quantum query complexity lower bounds for graph properties
- `D2_fine_grained_quantum_complexity` — Fine-grained quantum complexity for linear algebra primitives
- `D3_random_circuit_sampling_hardness` — Average-case hardness and random circuit sampling

## 5) Quantum Information Theory: Inequalities, Convexity, Geometry

- `E1_entropic_inequalities` — New entropic inequalities or equality characterizations
- `E2_resource_theories_convex_geometry` — Resource theories as convex geometry
- `E3_nonlocal_games_operator_algebras` — Nonlocal games and operator-algebraic relaxations

## 6) Topology, Algebra, and Categorical Approaches

- `F1_braids_mapping_class_groups` — Braids, mapping class groups, and compilation metrics
- `F2_stabilizer_symplectic_geometry` — Stabilizer formalism via symplectic geometry
- `F3_categorical_semantics_completeness` — Category-theoretic semantics with completeness results

---

## First-Paper Strategy

1. **Define a model** (noise model / circuit class / graph restriction / input oracle model)
2. **State a main theorem** you can realistically prove in 3–12 pages
3. Add:
   - 1–2 illustrative examples
   - 1 comparison theorem ("this improves/extends X under assumption Y")
   - Optional small experiments (only if they support the math)

## Currently Active Areas

- **Quantum LDPC constructions/limits**
- **Tensor networks as rigorous simulation/complexity tool**
- **Physically constrained Hamiltonian complexity**
