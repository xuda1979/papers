"""Archival helper script retained as a no-op.

Earlier revisions of this repository used this filename for generator scripts that
inserted unsupported proof-completion claims into the Yang--Mills manuscript.
Those claims have been retracted. This file is intentionally inert so future
searches and accidental executions do not resurrect stale overclaiming text.

Current audited status: theorem_boundary.
Canonical status artifacts:
- split/app190_final_synthesis_mass_gap.tex
- split/app151_hamiltonian_identification_theorem_boundary.tex
- verfification/proof_state.json
"""

from __future__ import annotations


def main() -> int:
    print(
        "Archived no-op: repository status is theorem_boundary; "
        "no Clay-level Yang--Mills mass-gap proof is certified here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
