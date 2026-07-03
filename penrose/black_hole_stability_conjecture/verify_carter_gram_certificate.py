#!/usr/bin/env python3
"""Verify exact rational Gram-matrix SOS certificates for Carter charts."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_sos_identity as sos


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def parse_matrix(raw: Any, path: str) -> list[list[Fraction]]:
    require(isinstance(raw, list) and raw, f"{path} must be a nonempty matrix")
    matrix: list[list[Fraction]] = []
    width = None
    for row_index, row in enumerate(raw):
        require(isinstance(row, list) and row, f"{path}[{row_index}] must be a nonempty row")
        if width is None:
            width = len(row)
        require(len(row) == width, f"{path}[{row_index}] has inconsistent width")
        matrix.append([sos.parse_fraction(value) for value in row])
    require(len(matrix) == width, f"{path} must be square")
    return matrix


def determinant(matrix: list[list[Fraction]]) -> Fraction:
    size = len(matrix)
    if size == 0:
        return Fraction(1)
    work = [row[:] for row in matrix]
    det = Fraction(1)
    sign = 1
    for pivot_index in range(size):
        pivot_row = None
        for row_index in range(pivot_index, size):
            if work[row_index][pivot_index] != 0:
                pivot_row = row_index
                break
        if pivot_row is None:
            return Fraction(0)
        if pivot_row != pivot_index:
            work[pivot_index], work[pivot_row] = work[pivot_row], work[pivot_index]
            sign *= -1
        pivot = work[pivot_index][pivot_index]
        det *= pivot
        for row_index in range(pivot_index + 1, size):
            factor = work[row_index][pivot_index] / pivot
            if factor == 0:
                continue
            for col_index in range(pivot_index, size):
                work[row_index][col_index] -= factor * work[pivot_index][col_index]
    return det * sign


def principal_minor(matrix: list[list[Fraction]], indices: tuple[int, ...]) -> Fraction:
    return determinant([[matrix[row][col] for col in indices] for row in indices])


def validate_psd(matrix: list[list[Fraction]], path: str) -> list[dict[str, str]]:
    size = len(matrix)
    for row in range(size):
        for col in range(size):
            require(matrix[row][col] == matrix[col][row], f"{path} must be symmetric")

    checked = []
    for order in range(1, size + 1):
        for indices in itertools.combinations(range(size), order):
            minor = principal_minor(matrix, indices)
            require(minor >= 0, f"{path} principal minor {indices} is negative: {minor}")
            checked.append({"indices": ",".join(str(index) for index in indices), "determinant": str(minor)})
    return checked


def gram_polynomial(
    variables: tuple[str, ...],
    basis: list[str],
    matrix: list[list[Fraction]],
) -> sos.Polynomial:
    require(len(basis) == len(matrix), "basis length must match Gram matrix size")
    basis_polys = [sos.eval_expr({"poly": {monomial: "1"}}, variables) for monomial in basis]
    total = sos.Polynomial.zero(variables)
    for row, left in enumerate(basis_polys):
        for col, right in enumerate(basis_polys):
            coeff = matrix[row][col]
            if coeff == 0:
                continue
            total = total + (left * right).scale(coeff)
    return total


def parse_basis(raw: Any, variables: tuple[str, ...]) -> list[str]:
    require(isinstance(raw, list) and raw, "monomial_basis must be a nonempty list")
    seen = set()
    basis = []
    for index, monomial in enumerate(raw):
        require(isinstance(monomial, str) and monomial.strip(), f"monomial_basis[{index}] must be a nonempty string")
        parsed = sos.parse_monomial_key(monomial, variables)
        require(parsed not in seen, f"monomial_basis[{index}] duplicates an earlier monomial")
        seen.add(parsed)
        basis.append(monomial)
    return basis


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    variables_raw = data.get("variables")
    require(isinstance(variables_raw, list) and all(isinstance(v, str) for v in variables_raw), "variables must be a list of strings")
    require(len(variables_raw) == len(set(variables_raw)), "variables must be distinct")
    variables = tuple(variables_raw)
    basis = parse_basis(data.get("monomial_basis"), variables)
    matrix = parse_matrix(data.get("gram_matrix"), "gram_matrix")
    require(len(matrix) == len(basis), "gram_matrix size must match monomial_basis length")
    minors = validate_psd(matrix, "gram_matrix")

    target_raw = data.get("target_polynomial")
    require(isinstance(target_raw, dict), "target_polynomial must be an object")
    target = sos.parse_sparse_poly(target_raw, variables)
    gram = gram_polynomial(variables, basis, matrix)
    residual = gram - target
    require(not residual.terms, f"Gram identity failed with residual {residual.to_sparse_json()}")

    return {
        "accepted": True,
        "variables": list(variables),
        "basis_size": len(basis),
        "checked_principal_minors": len(minors),
        "principal_minors": minors,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify(load_json(args.certificate))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Gram certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
