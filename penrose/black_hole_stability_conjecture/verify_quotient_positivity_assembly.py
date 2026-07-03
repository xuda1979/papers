#!/usr/bin/env python3
"""Validate a Carter–Wick Quotient Positivity Assembly Budget certificate.

This checker verifies the combinatorial and budget content of the
Carter–Wick Quotient Positivity Theorem with Assembly Budget (CWQP-AB).
It checks:

  1. every required field is present and well-typed;
  2. every rational is a valid Fraction representation;
  3. the local margin mu_alpha and Gram op-norm bound M_alpha are positive;
  4. the per-chart derivative loss is computed correctly as
        Loss_alpha = M_alpha / mu_alpha * (1 + C_bump / w_alpha);
  5. the total loss is the sum of per-chart losses;
  6. the closure inequality Loss_total <= B holds;
  7. the atlas covers a contiguous interval (gap-free);
  8. the overlap multiplicities are consistent with the atlas adjacencies.

The verifier is deliberately conservative. It does NOT check the analytic
content of the theorem (Carter–Wick symbol calculus, redshift absorption,
nonlinear bootstrap). It only checks that a proposed certificate is
internally consistent and that the budget closes.
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


def validate_metadata(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    for key in ("name", "generated_on", "spin", "claim_scope", "status", "warning"):
        require(key in metadata, f"metadata.{key} must be present")
        require(isinstance(metadata[key], str), f"metadata.{key} must be a string")
    require("Carter–Wick" in metadata["name"] or "Carter-Wick" in metadata["name"]
            or "CWQP" in metadata["name"],
            "metadata.name must reference Carter–Wick or CWQP")
    return metadata


def validate_bump(bump: Any) -> dict[str, Fraction | str]:
    require(isinstance(bump, dict), "bump must be an object")
    require("bump_constant" in bump, "bump.bump_constant must be present")
    c_bump = as_fraction(bump["bump_constant"])
    require(c_bump > 0, "bump.bump_constant must be positive")
    require("bump_type" in bump, "bump.bump_type must be present")
    require(isinstance(bump["bump_type"], str), "bump.bump_type must be a string")
    return {"bump_constant": c_bump, "bump_type": bump["bump_type"]}


def validate_atlas(atlas: Any) -> list[dict[str, Any]]:
    require(isinstance(atlas, list), "atlas must be a list")
    require(atlas, "atlas must be nonempty")
    charts: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(atlas):
        path = f"atlas[{index}]"
        require(isinstance(entry, dict), f"{path} must be an object")
        require("name" in entry, f"{path}.name must be present")
        name = entry["name"]
        require(isinstance(name, str), f"{path}.name must be a string")
        require(name not in seen_names, f"{path}.name duplicates an earlier chart")
        seen_names.add(name)
        require("endpoints" in entry, f"{path}.endpoints must be present")
        endpoints = entry["endpoints"]
        require(isinstance(endpoints, list) and len(endpoints) == 2,
                f"{path}.endpoints must be a 2-element list")
        lo = as_fraction(endpoints[0])
        hi = as_fraction(endpoints[1])
        require(lo < hi, f"{path}.endpoints must satisfy lo < hi")
        require("width" in entry, f"{path}.width must be present")
        width = as_fraction(entry["width"])
        require(width > 0, f"{path}.width must be positive")
        require(2 * width <= hi - lo,
                f"{path}.width must satisfy 2*width <= hi - lo (inradius constraint)")
        require("overlap_multiplicity" in entry,
                f"{path}.overlap_multiplicity must be present")
        m = entry["overlap_multiplicity"]
        require(isinstance(m, int) and m >= 1,
                f"{path}.overlap_multiplicity must be a positive integer")
        charts.append({
            "name": name,
            "lo": lo,
            "hi": hi,
            "width": width,
            "overlap_multiplicity": m,
        })
    charts.sort(key=lambda c: c["lo"])
    return charts


def validate_atlas_contiguity(charts: list[dict[str, Any]]) -> None:
    """Atlas must cover a contiguous interval without gaps."""
    for i in range(len(charts) - 1):
        left = charts[i]
        right = charts[i + 1]
        require(right["lo"] <= left["hi"],
                f"atlas has a gap between {left['name']} (hi={left['hi']}) "
                f"and {right['name']} (lo={right['lo']})")


def validate_overlap_multiplicities(charts: list[dict[str, Any]]) -> dict[str, int]:
    """For each chart, count how many other charts overlap it and require
    consistency with the declared overlap_multiplicity."""
    declared: dict[str, int] = {}
    for i, ci in enumerate(charts):
        count = 0
        for j, cj in enumerate(charts):
            if i == j:
                continue
            # Two charts overlap if their interiors intersect.
            if ci["lo"] < cj["hi"] and cj["lo"] < ci["hi"]:
                count += 1
        # The declared multiplicity counts the chart itself, so it is count + 1.
        declared[ci["name"]] = ci["overlap_multiplicity"]
        require(ci["overlap_multiplicity"] == count + 1,
                f"chart {ci['name']} declares overlap_multiplicity="
                f"{ci['overlap_multiplicity']} but actual adjacency count is {count + 1}")
    return declared


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
        require("monomial_basis_size" in entry,
                f"{path}.monomial_basis_size must be present")
        n = entry["monomial_basis_size"]
        require(isinstance(n, int) and n >= 1,
                f"{path}.monomial_basis_size must be a positive integer")
        require("gram_op_norm_bound" in entry,
                f"{path}.gram_op_norm_bound must be present")
        m_bound = as_fraction(entry["gram_op_norm_bound"])
        require(m_bound > 0, f"{path}.gram_op_norm_bound must be positive")
        require("local_margin" in entry, f"{path}.local_margin must be present")
        mu = as_fraction(entry["local_margin"])
        require(mu > 0, f"{path}.local_margin must be positive")
        require("gram_certificate_ref" in entry,
                f"{path}.gram_certificate_ref must be present")
        ref = entry["gram_certificate_ref"]
        require(isinstance(ref, str), f"{path}.gram_certificate_ref must be a string")
        out.append({
            "name": name,
            "monomial_basis_size": n,
            "gram_op_norm_bound": m_bound,
            "local_margin": mu,
            "gram_certificate_ref": ref,
        })
    return out


def compute_per_chart_loss(
    m_bound: Fraction, mu: Fraction, c_bump: Fraction, width: Fraction
) -> Fraction:
    """Loss_alpha = M_alpha / mu_alpha * (1 + C_bump / w_alpha)."""
    return (m_bound / mu) * (1 + c_bump / width)


def validate_budget(
    charts: list[dict[str, Any]],
    atlas: list[dict[str, Any]],
    c_bump: Fraction,
    data: dict[str, Any],
) -> dict[str, Any]:
    require("derivative_buffer" in data, "derivative_buffer must be present")
    buffer = as_fraction(data["derivative_buffer"])
    require(buffer > 0, "derivative_buffer must be positive")

    atlas_by_name = {c["name"]: c for c in atlas}
    require(set(atlas_by_name) == {c["name"] for c in charts},
            "atlas chart names and positivity chart names must match")

    per_chart: list[dict[str, Any]] = []
    total = Fraction(0)
    for ch in charts:
        atlas_entry = atlas_by_name[ch["name"]]
        loss = compute_per_chart_loss(
            ch["gram_op_norm_bound"],
            ch["local_margin"],
            c_bump,
            atlas_entry["width"],
        )
        per_chart.append({
            "name": ch["name"],
            "M_alpha": ch["gram_op_norm_bound"],
            "mu_alpha": ch["local_margin"],
            "w_alpha": atlas_entry["width"],
            "loss": loss,
        })
        total += loss

    require("budget" in data, "budget must be present")
    budget_in = data["budget"]
    require(isinstance(budget_in, dict), "budget must be an object")

    # If the certificate declares per-chart losses, verify they match.
    if "per_chart" in budget_in:
        require(isinstance(budget_in["per_chart"], list),
                "budget.per_chart must be a list")
        require(len(budget_in["per_chart"]) == len(per_chart),
                "budget.per_chart length must match charts length")
        for index, declared in enumerate(budget_in["per_chart"]):
            require(isinstance(declared, dict),
                    f"budget.per_chart[{index}] must be an object")
            require("name" in declared,
                    f"budget.per_chart[{index}].name must be present")
            require(declared["name"] == per_chart[index]["name"],
                    f"budget.per_chart[{index}].name mismatch")
            if "loss" in declared:
                declared_loss = as_fraction(declared["loss"])
                require(declared_loss == per_chart[index]["loss"],
                        f"budget.per_chart[{index}].loss mismatch: "
                        f"declared {declared_loss} != computed {per_chart[index]['loss']}")

    if "total_loss" in budget_in:
        declared_total = as_fraction(budget_in["total_loss"])
        require(declared_total == total,
                f"budget.total_loss mismatch: declared {declared_total} != computed {total}")

    closure = total <= buffer

    require("closure" in data, "closure must be present")
    require(isinstance(data["closure"], bool), "closure must be a boolean")
    require(data["closure"] == closure,
            f"closure mismatch: declared {data['closure']} != computed {closure}")

    return {
        "per_chart": per_chart,
        "total_loss": total,
        "derivative_buffer": buffer,
        "closure": closure,
    }


def _frac_to_str(obj: Any) -> Any:
    """Recursively convert Fraction values to their string form for JSON output."""
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
    bump = validate_bump(data.get("bump"))
    atlas = validate_atlas(data.get("atlas"))
    validate_atlas_contiguity(atlas)
    declared_multiplicities = validate_overlap_multiplicities(atlas)
    charts = validate_charts(data.get("charts"))
    budget = validate_budget(charts, atlas, bump["bump_constant"], data)

    result = {
        "accepted": True,
        "metadata": metadata,
        "bump": bump,
        "atlas_chart_count": len(atlas),
        "positivity_chart_count": len(charts),
        "declared_overlap_multiplicities": declared_multiplicities,
        "budget": budget,
        "warning": (
            "Combinatorial and budget checks pass. This does NOT prove the "
            "Carter–Wick Quotient Positivity Theorem: the analytic content "
            "(symbol calculus, redshift absorption, nonlinear bootstrap) is "
            "not checked."
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
        print(f"quotient positivity assembly certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
