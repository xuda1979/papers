"""
Standard quantum error-correcting codes, specified exactly as stabilizer
generators so the harness can *derive* their [[n,k,d]] parameters rather than
trust memorised tables.
"""
from .stabilizer import StabilizerCode


def generic_stabilizer_code(generators: list[str], name: str = "code") -> StabilizerCode:
    return StabilizerCode(generators, name=name)


def ShorCode() -> StabilizerCode:
    """[[9,1,3]] Shor code = concatenation of bit-flip and phase-flip."""
    gens = [
        # phase-flip (Z) stabilizers across blocks
        "ZZIIIIIII", "IZZIIIIII",
        "IIIZZIIII", "IIIIZZIII",
        "IIIIIIZZI", "IIIIIIIZZ",
        # bit-flip (X) stabilizers within blocks
        "XXXXXXIII", "IIIXXXXXX",
    ]
    return StabilizerCode(gens, name="Shor")


def SteaneCode() -> StabilizerCode:
    """[[7,1,3]] Steane code (CSS from the Hamming [7,4,3] check matrix H).

    H = [0 0 0 1 1 1 1; 0 1 1 0 0 1 1; 1 0 1 0 1 0 1]
    X-type stabilizers = rows of H as X operators; Z-type = same rows as Z.
    """
    gens = [
        # X-type stabilizers (rows of H as X operators)
        "IIIXXXX",
        "IXXIIXX",
        "XIXIXIX",
        # Z-type stabilizers (same rows of H as Z operators)
        "IIIZZZZ",
        "IZZIIZZ",
        "ZIZIZIZ",
    ]
    return StabilizerCode(gens, name="Steane")


def FiveQubitCode() -> StabilizerCode:
    """[[5,1,3]] perfect code."""
    gens = [
        "XZZXI", "IXZZX", "XIXZZ", "ZXIXZ",
    ]
    return StabilizerCode(gens, name="FiveQubit")


def BitFlipCode() -> StabilizerCode:
    """[[3,1,1]] repetition code protecting against bit flips (distance ~1 here as a
    raw non-CSS stabilizer treatment; re-expressed for clarity)."""
    return StabilizerCode(["ZZI", "IZZ"], name="BitFlipRepetition")


def PhaseFlipCode() -> StabilizerCode:
    return StabilizerCode(["XXI", "IXX"], name="PhaseFlipRepetition")
