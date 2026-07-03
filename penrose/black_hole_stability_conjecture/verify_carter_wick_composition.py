#!/usr/bin/env python3
"""Validate a Carter–Wick Composition Calculus (CWCC) certificate.

This checker verifies the arithmetic content of the Carter–Wick
Composition Calculus theorem (CWCC). It checks:

  1. every required field is present, well-typed, and rational;
  2. kappa_CW > 0;
  3. each chart has N_alpha >= 1, m_alpha >= 1, M_alpha > 0,
     mu_alpha > 0, w_alpha > 0;
  4. rho_alpha = kappa_CW * N_alpha^2 * m_alpha^2 * M_alpha *
     (1 + C_bump / w_alpha);
  5. Loss_alpha_comp = rho_alpha / mu_alpha;
  6. Loss_alpha_v2 = (M_alpha / mu_alpha)(1 + C_bump/w_alpha) +
     Loss_alpha_comp;
  7. total_loss_v2 = sum_alpha Loss_alpha_v2;
  8. if rrbb_compatibility is present, total_loss_v2 <= B_lower_bound.

The verifier is deliberately conservative. It does NOT check the
analytic content of the theorem (diagonal symplectic form, Moyal
expansion termination, Sobolev embedding). It only checks that a
proposed certificate is internally consistent and that the
composition-aware loss budget closes.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


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
        "Carter–Wick" in metadata["name"] or "Carter-Wick" in metadata["name"]
        or "CWCC" in metadata["name"],
        "metadata.name must reference Carter–Wick or CWCC",
    )
    return metadata


def validate_kappa_cw(data: dict[str, Any]) -> Fraction:
    require("carter_diagonal_constant" in data,
            "carter_diagonal_constant must be present")
    return as_positive_fraction(data["carter_diagonal_constant"],
                                "carter_diagonal_constant")


def validate_bump(data: dict[str, Any]) -> Fraction:
    bump = data.get("bump")
    require(isinstance(bump, dict), "bump must be an object")
    require("bump_constant" in bump, "bump.bump_constant must be present")
    return as_positive_fraction(bump["bump_constant"], "bump.bump_constant")


def validate_charts(charts_in: Any) -> list[dict[str, Any]]:
    require(isinstance(charts_in, list), "charts must be a list")
    require(charts_in, "charts must be nonempty")
    out: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(charts_in):
        path = f"charts[{index}]"
        require(isinstance(entry, dict), f"{path} must be an object")
        require("name" in entry, f"{path}.name must be present")
        name = entry["name"]
        require(isinstance(name, str), f"{path}.name must be a string")
        require(name not in seen_names, f"{path}.name duplicates an earlier chart")
        seen_names.add(name)
        # N_alpha
        require("monomial_basis_size" in entry,
                f"{path}.monomial_basis_size must be present")
        n_basis = entry["monomial_basis_size"]
        require(isinstance(n_basis, int) and n_basis >= 1,
                f"{path}.monomial_basis_size must be a positive integer")
        # m_alpha
        require("max_symbol_order" in entry,
                f"{path}.max_symbol_order must be present")
        m_order = entry["max_symbol_order"]
        require(isinstance(m_order, int) and m_order >= 1,
                f"{path}.max_symbol_order must be a positive integer")
        # M_alpha, mu_alpha, w_alpha
        m_bound = as_positive_fraction(entry.get("gram_op_norm_bound"),
                                        f"{path}.gram_op_norm_bound")
        mu = as_positive_fraction(entry.get("local_margin"),
                                   f"{path}.local_margin")
        w = as_positive_fraction(entry.get("width"), f"{path}.width")
        out.append({
            "name": name,
            "N_alpha": n_basis,
            "m_alpha": m_order,
            "M_alpha": m_bound,
            "mu_alpha": mu,
            "w_alpha": w,
        })
    return out


def compute_rho(
    kappa_cw: Fraction, n_alpha: int, m_alpha: int,
    m_bound: Fraction, c_bump: Fraction, w_alpha: Fraction,
) -> Fraction:
    """rho_alpha = kappa_CW * N^2 * m^2 * M * (1 + C_bump/w)."""
    return (
        kappa_cw
        * (n_alpha ** 2)
        * (m_alpha ** 2)
        * m_bound
        * (1 + c_bump / w_alpha)
    )


def compute_loss_v1(
    m_bound: Fraction, mu: Fraction, c_bump: Fraction, w_alpha: Fraction
) -> Fraction:
    """Loss_alpha^(1) = (M/mu) * (1 + C_bump/w)."""
    return (m_bound / mu) * (1 + c_bump / w_alpha)


def compute_loss_comp(rho: Fraction, mu: Fraction) -> Fraction:
    """Loss_alpha_comp = rho / mu."""
    return rho / mu


def compute_loss_v2(loss_v1: Fraction, loss_comp: Fraction) -> Fraction:
    """Loss_alpha^(2) = Loss^(1) + Loss_comp."""
    return loss_v1 + loss_comp


def validate_composition_loss(
    data: dict[str, Any],
    charts: list[dict[str, Any]],
    kappa_cw: Fraction,
    c_bump: Fraction,
) -> dict[str, Any]:
    comp_loss = data.get("composition_loss")
    require(isinstance(comp_loss, dict), "composition_loss must be an object")
    require("per_chart" in comp_loss,
            "composition_loss.per_chart must be present")
    per_chart_in = comp_loss["per_chart"]
    require(isinstance(per_chart_in, list),
            "composition_loss.per_chart must be a list")
    require(len(per_chart_in) == len(charts),
            "composition_loss.per_chart length must match charts length")

    per_chart: list[dict[str, Any]] = []
    total_v2 = Fraction(0)
    for index, declared in enumerate(per_chart_in):
        require(isinstance(declared, dict),
                f"composition_loss.per_chart[{index}] must be an object")
        require("name" in declared,
                f"composition_loss.per_chart[{index}].name must be present")
        require(declared["name"] == charts[index]["name"],
                f"composition_loss.per_chart[{index}].name mismatch")
        ch = charts[index]
        rho = compute_rho(
            kappa_cw, ch["N_alpha"], ch["m_alpha"],
            ch["M_alpha"], c_bump, ch["w_alpha"],
        )
        loss_comp = compute_loss_comp(rho, ch["mu_alpha"])
        loss_v1 = compute_loss_v1(
            ch["M_alpha"], ch["mu_alpha"], c_bump, ch["w_alpha"]
        )
        loss_v2 = compute_loss_v2(loss_v1, loss_comp)
        # Verify any declared values match.
        if "rho_alpha" in declared:
            declared_rho = as_nonnegative_fraction(declared["rho_alpha"],
                                                    f"composition_loss.per_chart[{index}].rho_alpha")
            require(declared_rho == rho,
                    f"composition_loss.per_chart[{index}].rho_alpha mismatch: "
                    f"declared {declared_rho} != computed {rho}")
        if "loss_comp" in declared:
            declared_lc = as_nonnegative_fraction(declared["loss_comp"],
                                                   f"composition_loss.per_chart[{index}].loss_comp")
            require(declared_lc == loss_comp,
                    f"composition_loss.per_chart[{index}].loss_comp mismatch: "
                    f"declared {declared_lc} != computed {loss_comp}")
        if "loss_v2" in declared:
            declared_lv2 = as_positive_fraction(declared["loss_v2"],
                                                 f"composition_loss.per_chart[{index}].loss_v2")
            require(declared_lv2 == loss_v2,
                    f"composition_loss.per_chart[{index}].loss_v2 mismatch: "
                    f"declared {declared_lv2} != computed {loss_v2}")
        per_chart.append({
            "name": ch["name"],
            "rho_alpha": rho,
            "loss_comp": loss_comp,
            "loss_v1": loss_v1,
            "loss_v2": loss_v2,
        })
        total_v2 += loss_v2

    if "total_loss_v2" in comp_loss:
        declared_total = as_positive_fraction(comp_loss["total_loss_v2"],
                                               "composition_loss.total_loss_v2")
        require(declared_total == total_v2,
                f"composition_loss.total_loss_v2 mismatch: declared "
                f"{declared_total} != computed {total_v2}")

    return {"per_chart": per_chart, "total_loss_v2": total_v2}


def validate_rrbb_compatibility(
    data: dict[str, Any], total_loss_v2: Fraction
) -> dict[str, Any] | None:
    rrbb = data.get("rrbb_compatibility")
    if rrbb is None:
        return None
    require(isinstance(rrbb, dict), "rrbb_compatibility must be an object")
    require("rrbb_certificate_ref" in rrbb,
            "rrbb_compatibility.rrbb_certificate_ref must be present")
    ref = rrbb["rrbb_certificate_ref"]
    require(isinstance(ref, str) and ref,
            "rrbb_compatibility.rrbb_certificate_ref must be a nonempty string")
    require("B_lower_bound" in rrbb,
            "rrbb_compatibility.B_lower_bound must be present")
    b_lb = as_positive_fraction(rrbb["B_lower_bound"],
                                 "rrbb_compatibility.B_lower_bound")
    closure = total_loss_v2 <= b_lb
    require("closure" in rrbb, "rrbb_compatibility.closure must be present")
    require(isinstance(rrbb["closure"], bool),
            "rrbb_compatibility.closure must be a boolean")
    require(rrbb["closure"] == closure,
            f"rrbb_compatibility.closure mismatch: declared {rrbb['closure']} "
            f"!= computed {closure} (total_loss_v2={total_loss_v2}, "
            f"B_lower_bound={b_lb})")
    return {
        "rrbb_certificate_ref": ref,
        "B_lower_bound": b_lb,
        "total_loss_v2": total_loss_v2,
        "closure": closure,
    }


def _frac_to_str(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _frac_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_frac_to_str(v) for v in obj]
    return obj


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    metadata = validate_metadata(data)
    kappa_cw = validate_kappa_cw(data)
    c_bump = validate_bump(data)
    charts = validate_charts(data.get("charts"))
    comp_loss = validate_composition_loss(data, charts, kappa_cw, c_bump)
    rrbb_compat = validate_rrbb_compatibility(data, comp_loss["total_loss_v2"])

    result = {
        "accepted": True,
        "metadata": metadata,
        "carter_diagonal_constant": kappa_cw,
        "bump_constant": c_bump,
        "chart_count": len(charts),
        "composition_loss": comp_loss,
        "rrbb_compatibility": rrbb_compat,
        "warning": (
            "Arithmetic checks pass. This does NOT prove the Carter–Wick "
            "Composition Calculus theorem: the analytic content (diagonal "
            "symplectic form, Moyal expansion termination, Sobolev "
            "embedding) is not checked."
        ),
    }
    return _frac_to_str(result)


def verify_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return verify(data)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except ValueError as exc:
        print(f"CWCC certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
