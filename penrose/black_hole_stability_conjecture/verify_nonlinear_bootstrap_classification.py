#!/usr/bin/env python3
"""Validate a Nonlinear Bootstrap Classification Theorem (NBCT) certificate.

This checker verifies the combinatorial and arithmetic content of the
Nonlinear Bootstrap Classification Theorem (NBCT). It checks:

  1. every required field is present, well-typed, and rational;
  2. every source term has a valid class label (NF, RS, IR) and positive
     rational coefficient;
  3. B_NF = sum_{i in NF} C_i;
  4. B_RS = sum_{i in RS} C_i * kappa_red * mu_redshift, reading
     kappa_red, mu_redshift from the referenced RRBB certificate;
  5. B_IR = sum_{i in IR} C_i * tau_0^{-eta_i}, with integer eta_i for
     exact rationality;
  6. B_NL = B_NF + B_RS + B_IR;
  7. the bootstrap closure B_NL * epsilon_boot^2 <= B_lower_bound holds,
     accepting either the tight case (equality) or the slack case
     (a rational lower bound epsilon with B_NL * epsilon^2 <= B_lower_bound).

The verifier is deliberately conservative. It does NOT check the
analytic content of the theorem (source-term classification, linear-stack
constant, Bondi flux formula). It only checks that a proposed certificate
is internally consistent and that the bootstrap closure closes.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_redshift_rp_buffer as rrbb

VALID_CLASSES = {"NF", "RS", "IR"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_fraction(value: Any) -> Fraction:
    require(isinstance(value, str), f"rational must be a string, got {type(value).__name__}: {value!r}")
    try:
        return Fraction(value)
    except (ArithmeticError, ValueError) as exc:
        raise ValueError(f"invalid rational {value!r}: {exc}") from exc


def as_positive_fraction(value: Any, name: str) -> Fraction:
    f = as_fraction(value)
    require(f > 0, f"{name} must be positive, got {f}")
    return f


def as_nonnegative_fraction(value: Any, name: str) -> Fraction:
    f = as_fraction(value)
    require(f >= 0, f"{name} must be nonnegative, got {f}")
    return f


def validate_metadata(data: dict[str, Any]) -> dict[str, str]:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    for key in ("name", "generated_on", "spin", "claim_scope", "status", "warning"):
        require(key in metadata, f"metadata.{key} must be present")
        require(isinstance(metadata[key], str), f"metadata.{key} must be a string")
    require(
        "Nonlinear" in metadata["name"] or "Bootstrap" in metadata["name"]
        or "NBCT" in metadata["name"],
        "metadata.name must reference Nonlinear, Bootstrap, or NBCT",
    )
    return metadata


def validate_linear_stack_reference(
    data: dict[str, Any]
) -> dict[str, Any]:
    ref = data.get("linear_stack_reference")
    require(isinstance(ref, dict), "linear_stack_reference must be an object")
    require("rrbb_certificate_ref" in ref,
            "linear_stack_reference.rrbb_certificate_ref must be present")
    rrbb_ref = ref["rrbb_certificate_ref"]
    require(isinstance(rrbb_ref, str) and rrbb_ref,
            "linear_stack_reference.rrbb_certificate_ref must be a nonempty string")
    require("B_lower_bound" in ref,
            "linear_stack_reference.B_lower_bound must be present")
    b_lb = as_positive_fraction(ref["B_lower_bound"],
                                 "linear_stack_reference.B_lower_bound")
    # Optional: actually load and verify the RRBB certificate to confirm
    # the B_lower_bound matches. This makes NBCT compose tightly with RRBB.
    if "cwcc_certificate_ref" in ref and "cwqp_certificate_ref" in ref:
        # Just record the refs; full cross-verification is done in the
        # composed full-stack verifier.
        pass
    return {"rrbb_certificate_ref": rrbb_ref, "B_lower_bound": b_lb}


def validate_source_terms(
    source_terms: Any, tau_0: Fraction
) -> tuple[list[dict[str, Any]], dict[str, Fraction]]:
    require(isinstance(source_terms, list), "source_terms must be a list")
    require(source_terms, "source_terms must be nonempty")
    out: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(source_terms):
        path = f"source_terms[{index}]"
        require(isinstance(entry, dict), f"{path} must be an object")
        require("name" in entry, f"{path}.name must be present")
        name = entry["name"]
        require(isinstance(name, str) and name,
                f"{path}.name must be a nonempty string")
        require(name not in seen_names, f"{path}.name duplicates an earlier source term")
        seen_names.add(name)
        require("class" in entry, f"{path}.class must be present")
        cls = entry["class"]
        require(cls in VALID_CLASSES,
                f"{path}.class must be one of {sorted(VALID_CLASSES)}, got {cls!r}")
        require("coefficient" in entry, f"{path}.coefficient must be present")
        coeff = as_positive_fraction(entry["coefficient"], f"{path}.coefficient")
        record: dict[str, Any] = {
            "name": name,
            "class": cls,
            "coefficient": coeff,
        }
        if cls == "IR":
            require("decay_exponent" in entry,
                    f"{path}.decay_exponent must be present for IR class")
            eta = as_fraction(entry["decay_exponent"])
            require(eta > 0,
                    f"{path}.decay_exponent must be positive, got {eta}")
            # Require integer eta for exact rationality of tau_0^{-eta}.
            require(eta.denominator == 1,
                    f"{path}.decay_exponent must be an integer for exact "
                    f"rational tau_0^(-eta), got {eta}")
            record["decay_exponent"] = int(eta)
            record["weight"] = coeff * (tau_0 ** (-int(eta)))
        elif cls == "RS":
            # weight will be filled in once kappa_red, mu_redshift are known.
            record["weight"] = None  # type: ignore[assignment]
        elif cls == "NF":
            record["weight"] = coeff
        out.append(record)
    # Compute per-class budgets (RS weights need kappa_red, mu_redshift).
    # We return the records and let the caller fill in RS weights.
    return out, {}


def fill_rs_weights(
    records: list[dict[str, Any]], kappa_red: Fraction, mu_redshift: Fraction
) -> None:
    for record in records:
        if record["class"] == "RS":
            record["weight"] = record["coefficient"] * kappa_red * mu_redshift


def compute_budgets(records: list[dict[str, Any]]) -> dict[str, Fraction]:
    b_nf = sum(
        (r["weight"] for r in records if r["class"] == "NF"),
        Fraction(0),
    )
    b_rs = sum(
        (r["weight"] for r in records if r["class"] == "RS"),
        Fraction(0),
    )
    b_ir = sum(
        (r["weight"] for r in records if r["class"] == "IR"),
        Fraction(0),
    )
    b_nl = b_nf + b_rs + b_ir
    return {"B_NF": b_nf, "B_RS": b_rs, "B_IR": b_ir, "B_NL": b_nl}


def validate_nonlinear_budget(
    data: dict[str, Any], computed: dict[str, Fraction]
) -> dict[str, Fraction]:
    nb = data.get("nonlinear_budget")
    require(isinstance(nb, dict), "nonlinear_budget must be an object")
    declared: dict[str, Fraction] = {}
    for key in ("B_NF", "B_RS", "B_IR", "B_NL"):
        require(key in nb, f"nonlinear_budget.{key} must be present")
        declared[key] = as_nonnegative_fraction(nb[key], f"nonlinear_budget.{key}")
        require(declared[key] == computed[key],
                f"nonlinear_budget.{key} mismatch: declared {declared[key]} "
                f"!= computed {computed[key]}")
    require(declared["B_NL"] == declared["B_NF"] + declared["B_RS"] + declared["B_IR"],
            "nonlinear_budget.B_NL must equal B_NF + B_RS + B_IR")
    return declared


def validate_bootstrap_closure(
    data: dict[str, Any], b_nl: Fraction, b_lb: Fraction
) -> dict[str, Any]:
    bc = data.get("bootstrap_closure")
    require(isinstance(bc, dict), "bootstrap_closure must be an object")
    require("B_lower_bound" in bc,
            "bootstrap_closure.B_lower_bound must be present")
    declared_b_lb = as_positive_fraction(bc["B_lower_bound"],
                                          "bootstrap_closure.B_lower_bound")
    require(declared_b_lb == b_lb,
            f"bootstrap_closure.B_lower_bound mismatch: declared "
            f"{declared_b_lb} != linear_stack_reference {b_lb}")
    require("epsilon_boot" in bc,
            "bootstrap_closure.epsilon_boot must be present")
    eps = as_positive_fraction(bc["epsilon_boot"],
                                "bootstrap_closure.epsilon_boot")
    require("closure" in bc, "bootstrap_closure.closure must be present")
    require(isinstance(bc["closure"], bool),
            "bootstrap_closure.closure must be a boolean")
    # Check B_NL * eps^2 <= B_lower_bound.
    closure = b_nl * eps * eps <= b_lb
    require(bc["closure"] == closure,
            f"bootstrap_closure.closure mismatch: declared {bc['closure']} "
            f"!= computed {closure} (B_NL={b_nl}, eps={eps}, "
            f"B_NL*eps^2={b_nl * eps * eps}, B_lower_bound={b_lb})")
    return {
        "epsilon_boot": eps,
        "B_lower_bound": b_lb,
        "B_NL_eps_squared": b_nl * eps * eps,
        "closure": closure,
        "tight": b_nl * eps * eps == b_lb,
    }


def _frac_to_str(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _frac_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_frac_to_str(v) for v in obj]
    return obj


def verify(data: dict[str, Any], rrbb_path: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    metadata = validate_metadata(data)
    lin_ref = validate_linear_stack_reference(data)

    # Load kappa_red and mu_redshift from the referenced RRBB certificate.
    require(rrbb_path is not None,
            "rrbb_path must be provided to verify an NBCT certificate")
    rrbb_data = json.loads(rrbb_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    rrbb_result = rrbb.verify(rrbb_data)
    kappa_red = Fraction(rrbb_result["redshift_certificate"]["kappa_red"])
    mu_redshift = Fraction(rrbb_result["redshift_certificate"]["mu_redshift"])

    # Validate bootstrap parameters (need tau_0 before source terms).
    bp = data.get("bootstrap_parameters")
    require(isinstance(bp, dict), "bootstrap_parameters must be an object")
    require("tau_0" in bp, "bootstrap_parameters.tau_0 must be present")
    tau_0 = as_positive_fraction(bp["tau_0"], "bootstrap_parameters.tau_0")
    require(tau_0 >= 1,
            f"bootstrap_parameters.tau_0 must be >= 1 for decay, got {tau_0}")
    require("sobolev_index" in bp,
            "bootstrap_parameters.sobolev_index must be present")
    sobolev = bp["sobolev_index"]
    require(isinstance(sobolev, int) and sobolev >= 1,
            f"bootstrap_parameters.sobolev_index must be a positive integer")

    records, _ = validate_source_terms(data.get("source_terms"), tau_0)
    fill_rs_weights(records, kappa_red, mu_redshift)
    computed = compute_budgets(records)
    declared = validate_nonlinear_budget(data, computed)
    closure = validate_bootstrap_closure(
        data, computed["B_NL"], lin_ref["B_lower_bound"]
    )

    result = {
        "accepted": True,
        "metadata": metadata,
        "linear_stack_reference": lin_ref,
        "bootstrap_parameters": {"tau_0": tau_0, "sobolev_index": sobolev},
        "source_term_count": len(records),
        "source_terms_by_class": {
            "NF": sum(1 for r in records if r["class"] == "NF"),
            "RS": sum(1 for r in records if r["class"] == "RS"),
            "IR": sum(1 for r in records if r["class"] == "IR"),
        },
        "nonlinear_budget": declared,
        "bootstrap_closure": closure,
        "rrbb_constants": {
            "kappa_red": kappa_red,
            "mu_redshift": mu_redshift,
        },
        "warning": (
            "Combinatorial and arithmetic checks pass. This does NOT prove "
            "the Nonlinear Bootstrap Classification Theorem: the analytic "
            "content (source-term classification, linear-stack constant, "
            "Bondi flux formula) is not checked."
        ),
    }
    return _frac_to_str(result)


def verify_file(path: Path, rrbb_path: Path | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if rrbb_path is None:
        # Read the RRBB reference from the certificate to locate the file.
        ref = data.get("linear_stack_reference", {})
        rrbb_ref = ref.get("rrbb_certificate_ref")
        if rrbb_ref is not None:
            # Walk up from the certificate's directory to find the RRBB file.
            # The reference path is relative to the repo root.
            found: Path | None = None
            current = path.parent.resolve()
            while True:
                candidate = current / rrbb_ref
                if candidate.exists():
                    found = candidate
                    break
                if current.parent == current:
                    break  # reached filesystem root
                current = current.parent
            if found is None:
                raise ValueError(
                    f"could not locate RRBB certificate at {rrbb_ref}"
                )
            rrbb_path = found
    return verify(data, rrbb_path)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--rrbb", type=Path, default=None,
                        help="path to the referenced RRBB certificate "
                             "(default: resolve from certificate)")
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate, args.rrbb)
    except ValueError as exc:
        print(f"NBCT certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
