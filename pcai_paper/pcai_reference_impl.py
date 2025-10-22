from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Any, Dict, Tuple, List, Optional
import argparse
import hashlib
import json
import math
import sys
import time


JSONDict = Dict[str, Any]


# ------------------------------
# Certificate primitives
# ------------------------------
@dataclass(frozen=True)
class Certificate:
    """Machine-checkable certificate.

    Notes
    -----
    - `digest()` is computed from a *canonicalized* JSON body that excludes
      the volatile `verified_at` timestamp so identical proofs hash to the
      same digest across runs.
    - `to_json()` includes all fields for auditability (including `verified_at`).
    """
    kind: str
    payload: JSONDict
    verified_at: float
    verifier_version: str = "0.3.0"

    def _canonical_body(self) -> JSONDict:
        # Exclude `verified_at` from the digest to keep it deterministic.
        return {
            "kind": self.kind,
            "payload": self.payload,
            "verifier_version": self.verifier_version,
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def to_canonical_json(self) -> str:
        return json.dumps(self._canonical_body(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()


class VerificationError(RuntimeError):
    """Exception raised when a verification step fails.

    The (failed) certificate is attached so callers can inspect the payload.
    """

    def __init__(self, message: str, certificate: Certificate):
        super().__init__(message)
        self.certificate = certificate


def _stamp(kind: str, payload: JSONDict) -> Certificate:
    return Certificate(kind=kind, payload=payload, verified_at=time.time())


# ------------------------------
# Verifiers
# ------------------------------
def verify_forall_range(
    fn: Callable[[int], int],
    spec: Callable[[int], int],
    N: int,
) -> Tuple[bool, Certificate]:
    """Verify ∀ n ∈ [0..N]: ``fn(n) == spec(n)``.

    Returns (ok, certificate). The certificate contains the first
    counterexample (if any) and metadata to aid reproducibility.
    """
    if N < 0:
        raise ValueError("N must be non-negative")

    counterexample: Optional[JSONDict] = None
    fn_name = getattr(fn, "__name__", repr(fn))
    spec_name = getattr(spec, "__name__", repr(spec))

    for n in range(N + 1):
        try:
            lhs, rhs = fn(n), spec(n)
        except Exception as e:  # noqa: BLE001 - we capture and record
            counterexample = {"n": n, "error": f"{type(e).__name__}: {e}"}
            break
        if lhs != rhs:
            counterexample = {"n": n, "lhs": lhs, "rhs": rhs}
            break

    ok = counterexample is None
    payload: JSONDict = {
        "ok": ok,
        "N": N,
        "fn": fn_name,
        "spec": spec_name,
        "counterexample": counterexample,
    }
    return ok, _stamp("forall_range", payload)


def verify_divide_trace(
    a: float,
    b: float,
    result: float,
    *,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-9,
) -> Tuple[bool, Certificate]:
    """Contract-style verification for division.

    Preconditions
    -------------
    - b != 0
    - a and b are finite floats

    Postconditions
    --------------
    - result is finite
    - result ≈ a / b within provided tolerances
    """
    pre = {
        "a": a,
        "b": b,
        "require": "b != 0 and finite(a,b)",
    }
    pre_ok = math.isfinite(a) and math.isfinite(b) and (b != 0.0)
    if not pre_ok:
        violation = []
        if not math.isfinite(a):
            violation.append("non-finite a")
        if not math.isfinite(b):
            violation.append("non-finite b")
        if b == 0.0:
            violation.append("b == 0")
        payload = {"ok": False, "violation": ", ".join(violation), "pre": pre}
        return False, _stamp("contract_trace", payload)

    approx_ok = math.isfinite(result) and math.isclose(
        result, a / b, rel_tol=rel_tol, abs_tol=abs_tol
    )
    post = {
        "approx": f"math.isclose(result,a/b,rel_tol={rel_tol},abs_tol={abs_tol})",
        "finite": True,
    }
    payload = {"ok": approx_ok, "pre": pre, "post": post}
    return approx_ok, _stamp("contract_trace", payload)


# ------------------------------
# Tasks / specs
# ------------------------------
def triangular_spec(n: int) -> int:
    """Closed-form spec for sum_{i=0}^n i."""
    return n * (n + 1) // 2


def sum_first_n_candidate(n: int) -> int:
    """Candidate implementation (loop), decoupled from the spec."""
    total = 0
    for i in range(n + 1):
        total += i
    return total


def build_sum_first_n(N: int = 5000) -> Tuple[Callable[[int], int], Certificate]:
    """Returns a candidate function and a certificate that it matches the
    spec for all n in [0..N].
    """
    fn = sum_first_n_candidate
    ok, cert = verify_forall_range(fn, triangular_spec, N)
    if not ok:
        raise VerificationError(
            f"Failed to verify sum_first_n up to {N}: {cert.payload.get('counterexample')}",
            cert,
        )
    return fn, cert


def safe_divide(a: float, b: float) -> float:
    """Perform division (unsafe w.r.t. preconditions)."""
    return a / b


def wrapped_safe_divide(
    a: float, b: float, *, abs_tol: float = 1e-12, rel_tol: float = 1e-9
) -> Tuple[float, Certificate]:
    """Divide with contract checking and return (result, certificate)."""
    pre_ok = math.isfinite(a) and math.isfinite(b) and (b != 0.0)
    if not pre_ok:
        _, cert = verify_divide_trace(a, b, float("nan"), abs_tol=abs_tol, rel_tol=rel_tol)
        raise VerificationError(f"Division verification failed: {cert.payload}", cert)

    try:
        result = safe_divide(a, b)
    except ZeroDivisionError as exc:  # pragma: no cover - defensive
        _, cert = verify_divide_trace(a, b, float("nan"), abs_tol=abs_tol, rel_tol=rel_tol)
        raise VerificationError("Division raised ZeroDivisionError", cert) from exc

    ok, cert = verify_divide_trace(a, b, result, abs_tol=abs_tol, rel_tol=rel_tol)
    if not ok:
        raise VerificationError(f"Division verification failed: {cert.payload}", cert)
    return result, cert


# ------------------------------
# CLI / Demo
# ------------------------------
def _emit(tag: str, cert: Certificate, extra: str, mode: str = "text") -> None:
    if mode == "json":
        print(cert.to_json())
    elif mode == "digest":
        print(cert.digest())
    else:
        print(
            f"[{tag}] ACCEPT digest={cert.digest()} payload={cert.to_json()}{(' ' + extra) if extra else ''}"
        )


def _print_reject(tag: str, err: Exception) -> None:
    print(f"[{tag}] REJECT {err}", file=sys.stderr)


def demo(mode: str = "text") -> int:
    fn, cert = build_sum_first_n()
    _emit("sum_first_n", cert, "", mode=mode)

    try:
        out, cert2 = wrapped_safe_divide(7.0, 3.0)
        _emit("safe_divide", cert2, f"result={out}", mode=mode)
    except Exception as e:
        _print_reject("safe_divide", e)

    try:
        out, cert3 = wrapped_safe_divide(1.0, 0.0)
        _emit("safe_divide bad", cert3, f"result={out}", mode=mode)
    except Exception as e:
        _print_reject("safe_divide bad", e)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Proof-Carrying AI reference prototype")
    parser.add_argument(
        "--out",
        choices=("text", "json", "digest"),
        default="text",
        help="Output format for certificates",
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("demo", help="run the demo (sum + division)")

    p_sum = sub.add_parser("sum", help="verify sum_first_n up to N")
    p_sum.add_argument("--N", type=int, default=5000, help="Upper bound inclusive")

    p_div = sub.add_parser("divide", help="run verified division")
    p_div.add_argument("a", type=float)
    p_div.add_argument("b", type=float)
    p_div.add_argument("--abs-tol", type=float, default=1e-12, dest="abs_tol")
    p_div.add_argument("--rel-tol", type=float, default=1e-9, dest="rel_tol")

    args = parser.parse_args(argv)

    if args.cmd == "sum":
        _, cert = build_sum_first_n(args.N)
        _emit("sum_first_n", cert, "", mode=args.out)
        return 0

    if args.cmd == "divide":
        try:
            out, cert = wrapped_safe_divide(args.a, args.b, abs_tol=args.abs_tol, rel_tol=args.rel_tol)
            _emit("safe_divide", cert, f"result={out}", mode=args.out)
            return 0
        except Exception as e:
            _print_reject("safe_divide", e)
            return 2

    # default: demo
    return demo(mode=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
