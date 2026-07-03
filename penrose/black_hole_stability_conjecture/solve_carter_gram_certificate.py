#!/usr/bin/env python3
"""Construct small exact rational Gram certificates by square extraction.

The solver is deliberately modest. It searches for a rational decomposition of a
target polynomial as a nonnegative sum of squares of sparse integer linear forms
in the supplied monomial basis. The output is a Gram certificate that
``verify_carter_gram_certificate.py`` checks exactly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_gram_certificate as gram
import verify_carter_sos_identity as sos


@dataclass(frozen=True)
class CandidateSquare:
    vector: tuple[Fraction, ...]
    polynomial: sos.Polynomial
    support_size: int
    max_abs_coeff: Fraction
    has_cross_terms: bool


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def parse_variables(data: dict[str, Any]) -> tuple[str, ...]:
    variables = data.get("variables")
    require(isinstance(variables, list) and all(isinstance(v, str) for v in variables), "variables must be a list of strings")
    require(len(variables) == len(set(variables)), "variables must be distinct")
    return tuple(variables)


def basis_square_monomials(variables: tuple[str, ...], basis: list[str]) -> set[sos.Monomial]:
    result = set()
    for monomial in basis:
        poly = sos.eval_expr({"poly": {monomial: "1"}}, variables)
        square = poly * poly
        result.update(square.terms)
    return result


def normalize_integer_coefficients(coefficients: tuple[int, ...]) -> tuple[int, ...] | None:
    if all(coeff == 0 for coeff in coefficients):
        return None
    divisor = 0
    for coeff in coefficients:
        divisor = math.gcd(divisor, abs(coeff))
    normalized = tuple(coeff // divisor for coeff in coefficients)
    for coeff in normalized:
        if coeff != 0:
            if coeff < 0:
                normalized = tuple(-item for item in normalized)
            return normalized
    return None


def linear_form_polynomial(
    variables: tuple[str, ...],
    basis_polys: list[sos.Polynomial],
    vector: tuple[Fraction, ...],
) -> sos.Polynomial:
    total = sos.Polynomial.zero(variables)
    for coeff, basis_poly in zip(vector, basis_polys):
        if coeff:
            total = total + basis_poly.scale(coeff)
    return total


def generate_candidates(
    variables: tuple[str, ...],
    basis: list[str],
    max_coeff: int,
    max_support: int,
) -> list[CandidateSquare]:
    require(max_coeff >= 1, "max_coeff must be at least 1")
    require(max_support >= 1, "max_support must be at least 1")
    basis_polys = [sos.eval_expr({"poly": {monomial: "1"}}, variables) for monomial in basis]
    square_monomials = basis_square_monomials(variables, basis)
    seen: set[tuple[int, ...]] = set()
    candidates: list[CandidateSquare] = []
    size = len(basis)
    for support_size in range(1, min(max_support, size) + 1):
        for support in itertools.combinations(range(size), support_size):
            for raw_coeffs in itertools.product(range(-max_coeff, max_coeff + 1), repeat=support_size):
                normalized = normalize_integer_coefficients(raw_coeffs)
                if normalized is None:
                    continue
                vector_int = [0 for _ in basis]
                for index, coeff in zip(support, normalized):
                    vector_int[index] = coeff
                vector_key = tuple(vector_int)
                if vector_key in seen:
                    continue
                seen.add(vector_key)
                vector = tuple(Fraction(coeff) for coeff in vector_key)
                linear = linear_form_polynomial(variables, basis_polys, vector)
                square = linear * linear
                has_cross_terms = any(monomial not in square_monomials for monomial in square.terms)
                candidates.append(
                    CandidateSquare(
                        vector=vector,
                        polynomial=square,
                        support_size=sum(1 for coeff in vector if coeff),
                        max_abs_coeff=max(abs(coeff) for coeff in vector),
                        has_cross_terms=has_cross_terms,
                    )
                )
    return sorted(
        candidates,
        key=lambda candidate: (
            -int(candidate.has_cross_terms),
            -candidate.support_size,
            candidate.max_abs_coeff,
            sum(abs(coeff) for coeff in candidate.vector),
            candidate.vector,
        ),
    )


def positive_ratios(residual: sos.Polynomial, candidate: CandidateSquare) -> list[Fraction] | None:
    ratios = []
    for monomial, coeff in candidate.polynomial.terms.items():
        residual_coeff = residual.terms.get(monomial, Fraction(0))
        if residual_coeff == 0:
            return None
        ratio = residual_coeff / coeff
        if ratio <= 0:
            return None
        ratios.append(ratio)
    return ratios


def subtract_square(
    residual: sos.Polynomial,
    candidate: CandidateSquare,
    weight: Fraction,
) -> sos.Polynomial:
    return residual - candidate.polynomial.scale(weight)


def add_rank_one(matrix: list[list[Fraction]], vector: tuple[Fraction, ...], weight: Fraction) -> None:
    for row, left in enumerate(vector):
        if left == 0:
            continue
        for col, right in enumerate(vector):
            if right:
                matrix[row][col] += weight * left * right


def decomposition_entry(basis: list[str], candidate: CandidateSquare, weight: Fraction) -> dict[str, Any]:
    return {
        "weight": str(weight),
        "linear_form": {
            monomial: str(coeff)
            for monomial, coeff in zip(basis, candidate.vector)
            if coeff
        },
    }


def extract_squares(
    variables: tuple[str, ...],
    basis: list[str],
    target: sos.Polynomial,
    max_coeff: int,
    max_support: int,
    max_steps: int,
) -> tuple[list[list[Fraction]], list[dict[str, Any]]]:
    candidates = generate_candidates(variables, basis, max_coeff=max_coeff, max_support=max_support)
    residual = target
    matrix = [[Fraction(0) for _ in basis] for _ in basis]
    decomposition: list[dict[str, Any]] = []

    for _ in range(max_steps):
        if not residual.terms:
            return matrix, decomposition

        exact_choice: tuple[CandidateSquare, Fraction] | None = None
        for candidate in candidates:
            ratios = positive_ratios(residual, candidate)
            if not ratios:
                continue
            if all(ratio == ratios[0] for ratio in ratios):
                exact_choice = (candidate, ratios[0])
                break
        if exact_choice is not None:
            candidate, weight = exact_choice
            residual = subtract_square(residual, candidate, weight)
            add_rank_one(matrix, candidate.vector, weight)
            decomposition.append(decomposition_entry(basis, candidate, weight))
            continue

        partial_choice: tuple[CandidateSquare, Fraction] | None = None
        for candidate in candidates:
            ratios = positive_ratios(residual, candidate)
            if not ratios:
                continue
            weight = min(ratios)
            if weight > 0:
                partial_choice = (candidate, weight)
                break
        if partial_choice is None:
            break

        candidate, weight = partial_choice
        residual = subtract_square(residual, candidate, weight)
        add_rank_one(matrix, candidate.vector, weight)
        decomposition.append(decomposition_entry(basis, candidate, weight))

    if residual.terms:
        raise ValueError(f"failed to extract a nonnegative square decomposition; residual {residual.to_sparse_json()}")
    return matrix, decomposition


def matrix_to_json(matrix: list[list[Fraction]]) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def solve(
    data: dict[str, Any],
    max_coeff: int = 1,
    max_support: int = 3,
    max_steps: int = 64,
) -> dict[str, Any]:
    require(isinstance(data, dict), "input must be an object")
    variables = parse_variables(data)
    basis = gram.parse_basis(data.get("monomial_basis"), variables)
    target_raw = data.get("target_polynomial")
    require(isinstance(target_raw, dict), "target_polynomial must be an object")
    target = sos.parse_sparse_poly(target_raw, variables)
    matrix, decomposition = extract_squares(
        variables,
        basis,
        target,
        max_coeff=max_coeff,
        max_support=max_support,
        max_steps=max_steps,
    )
    result = {
        "metadata": {
            "name": data.get("metadata", {}).get("name", "solved Carter Gram certificate")
            if isinstance(data.get("metadata"), dict)
            else "solved Carter Gram certificate",
            "status": "solver-produced exact rational Gram certificate; not a Kerr proof by itself",
            "solver": "sparse rational square extraction",
            "max_coeff": max_coeff,
            "max_support": max_support,
        },
        "variables": list(variables),
        "monomial_basis": basis,
        "gram_matrix": matrix_to_json(matrix),
        "target_polynomial": {key: str(value) for key, value in target_raw.items()},
        "solver_decomposition": decomposition,
    }
    gram.verify(result)
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-coeff", type=int, default=1)
    parser.add_argument("--max-support", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=64)
    args = parser.parse_args(argv[1:])
    try:
        result = solve(
            load_json(args.input),
            max_coeff=args.max_coeff,
            max_support=args.max_support,
            max_steps=args.max_steps,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Gram solve failed: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
