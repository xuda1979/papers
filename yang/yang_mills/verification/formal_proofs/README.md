Formal Verification Roadmap
===========================

This directory contains the specifications and trace logs required to port the 
Python-based interval arithmetic verification to a Trusted Foundation (Lean 4 or Coq).

Structure
---------
1. `formal_spec.py`: A Python-syntax specification of the axioms and predicates 
   that must be proven. This serves as a "Z Specification" or "TLA+" style blueprint.

2. `trace_logs/`: (Planned) Directory to store execution traces from `shadow_flow_verifier.py`.
   The Python script acts as a "Prover" (finding the certificate), and the 
   Trace Checker (to be written in Lean) acts as the "Verifier".

Methodology
-----------
The "Kick-the-tires" approach for referees:
1. The Python script computes the bounds using `Interval` class.
2. The `Interval` class logic is simple and readable (see `interval_arithmetic.py`).
3. To remove reliance on the Python interpreter, one can export the sequence of 
   floating point operations and endpoints to a text file.
4. A small, verified kernel (written in Lean/C++) checks that `lower <= real_op <= upper` 
   for each step.

Status (Jan 2026)
-----------------
- Specification: DRAFT COMPLETE (`formal_spec.py`)
- Python Implementation: RIGOROUS (`interval_arithmetic.py` uses directed rounding)
- Trace Generation: PENDING (Implemented via `formal_verifier_interface`)
