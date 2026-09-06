"""Quantum error correction analysis: stabilizers, codes, distance, thresholds."""
from .stabilizer import (
    StabilizerCode, logical_encoding_ops, encode_circuit,
    distances_from_stabilizers, is_self_dual,
)
from .codes import (
    SteaneCode, ShorCode, BitFlipCode, PhaseFlipCode, FiveQubitCode, generic_stabilizer_code,
)
from .simulate import decode_syndrome, protective_circuit_fidelity

__all__ = [
    "StabilizerCode", "logical_encoding_ops", "encode_circuit",
    "distances_from_stabilizers", "is_self_dual",
    "SteaneCode", "ShorCode", "BitFlipCode", "PhaseFlipCode", "FiveQubitCode",
    "generic_stabilizer_code", "decode_syndrome", "protective_circuit_fidelity",
]
