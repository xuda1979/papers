#!/usr/bin/env python3
"""Verify exact rational polynomial identities for Carter SOS scaffolds.

This is a small exact-arithmetic checker. It verifies identities written in a
JSON expression grammar over sparse multivariate polynomials. It does not solve
for SOS coefficients and does not prove positivity of a Gram matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


Monomial = tuple[int, ...]


@dataclass(frozen=True)
class Polynomial:
    variables: tuple[str, ...]
    terms: dict[Monomial, Fraction]

    @staticmethod
    def zero(variables: tuple[str, ...]) -> "Polynomial":
        return Polynomial(variables, {})

    @staticmethod
    def constant(variables: tuple[str, ...], value: Fraction) -> "Polynomial":
        if value == 0:
            return Polynomial.zero(variables)
        return Polynomial(variables, {tuple(0 for _ in variables): value})

    @staticmethod
    def variable(variables: tuple[str, ...], name: str) -> "Polynomial":
        if name not in variables:
            raise ValueError(f"unknown variable {name!r}")
        powers = [0 for _ in variables]
        powers[variables.index(name)] = 1
        return Polynomial(variables, {tuple(powers): Fraction(1)})

    def _check(self, other: "Polynomial") -> None:
        if self.variables != other.variables:
            raise ValueError("polynomial variable mismatch")

    def __add__(self, other: "Polynomial") -> "Polynomial":
        self._check(other)
        terms = dict(self.terms)
        for monomial, coeff in other.terms.items():
            terms[monomial] = terms.get(monomial, Fraction(0)) + coeff
            if terms[monomial] == 0:
                del terms[monomial]
        return Polynomial(self.variables, terms)

    def __neg__(self) -> "Polynomial":
        return Polynomial(self.variables, {m: -c for m, c in self.terms.items()})

    def __sub__(self, other: "Polynomial") -> "Polynomial":
        return self + (-other)

    def __mul__(self, other: "Polynomial") -> "Polynomial":
        self._check(other)
        terms: dict[Monomial, Fraction] = {}
        for left_m, left_c in self.terms.items():
            for right_m, right_c in other.terms.items():
                monomial = tuple(a + b for a, b in zip(left_m, right_m))
                terms[monomial] = terms.get(monomial, Fraction(0)) + left_c * right_c
                if terms[monomial] == 0:
                    del terms[monomial]
        return Polynomial(self.variables, terms)

    def scale(self, value: Fraction) -> "Polynomial":
        if value == 0:
            return Polynomial.zero(self.variables)
        return Polynomial(self.variables, {m: value * c for m, c in self.terms.items()})

    def to_sparse_json(self) -> dict[str, str]:
        result = {}
        for monomial, coeff in sorted(self.terms.items()):
            key = "*".join(
                f"{var}^{power}" if power != 1 else var
                for var, power in zip(self.variables, monomial)
                if power
            )
            result[key or "1"] = str(coeff)
        return result


def parse_fraction(raw: Any) -> Fraction:
    try:
        return Fraction(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"invalid rational {raw!r}") from exc


def parse_monomial_key(key: str, variables: tuple[str, ...]) -> Monomial:
    powers = [0 for _ in variables]
    text = key.strip()
    if text in {"", "1"}:
        return tuple(powers)
    for factor in text.split("*"):
        if "^" in factor:
            name, raw_power = factor.split("^", 1)
            power = int(raw_power)
        else:
            name = factor
            power = 1
        if name not in variables:
            raise ValueError(f"unknown variable {name!r} in monomial {key!r}")
        if power < 0:
            raise ValueError("negative powers are not supported")
        powers[variables.index(name)] += power
    return tuple(powers)


def parse_sparse_poly(raw: dict[str, Any], variables: tuple[str, ...]) -> Polynomial:
    terms: dict[Monomial, Fraction] = {}
    for key, value in raw.items():
        coeff = parse_fraction(value)
        if coeff == 0:
            continue
        monomial = parse_monomial_key(key, variables)
        terms[monomial] = terms.get(monomial, Fraction(0)) + coeff
        if terms[monomial] == 0:
            del terms[monomial]
    return Polynomial(variables, terms)


def eval_expr(expr: Any, variables: tuple[str, ...]) -> Polynomial:
    if isinstance(expr, str):
        return Polynomial.variable(variables, expr)
    if isinstance(expr, int):
        return Polynomial.constant(variables, Fraction(expr))
    if isinstance(expr, dict):
        if "const" in expr:
            return Polynomial.constant(variables, parse_fraction(expr["const"]))
        if "var" in expr:
            return Polynomial.variable(variables, str(expr["var"]))
        if "poly" in expr:
            raw = expr["poly"]
            if not isinstance(raw, dict):
                raise ValueError("poly expression must be an object")
            return parse_sparse_poly(raw, variables)
        if "neg" in expr:
            return -eval_expr(expr["neg"], variables)
        if "sum" in expr:
            items = expr["sum"]
            if not isinstance(items, list):
                raise ValueError("sum expression must be a list")
            total = Polynomial.zero(variables)
            for item in items:
                total = total + eval_expr(item, variables)
            return total
        if "prod" in expr:
            items = expr["prod"]
            if not isinstance(items, list):
                raise ValueError("prod expression must be a list")
            total = Polynomial.constant(variables, Fraction(1))
            for item in items:
                total = total * eval_expr(item, variables)
            return total
        if "square" in expr:
            poly = eval_expr(expr["square"], variables)
            return poly * poly
        if "scale" in expr:
            spec = expr["scale"]
            if not isinstance(spec, dict) or "coeff" not in spec or "expr" not in spec:
                raise ValueError("scale expression must contain coeff and expr")
            return eval_expr(spec["expr"], variables).scale(parse_fraction(spec["coeff"]))
    raise ValueError(f"unsupported expression {expr!r}")


def verify(data: dict[str, Any]) -> dict[str, Any]:
    variables_raw = data.get("variables")
    if not isinstance(variables_raw, list) or not all(isinstance(v, str) for v in variables_raw):
        raise ValueError("variables must be a list of strings")
    variables = tuple(variables_raw)
    identities = data.get("identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("identities must be a nonempty list")

    checked = []
    for index, identity in enumerate(identities):
        if not isinstance(identity, dict):
            raise ValueError(f"identity {index} must be an object")
        name = identity.get("name", f"identity_{index}")
        if "lhs" not in identity or "rhs" not in identity:
            raise ValueError(f"{name}: identity must contain lhs and rhs")
        lhs = eval_expr(identity["lhs"], variables)
        rhs = eval_expr(identity["rhs"], variables)
        residual = lhs - rhs
        if residual.terms:
            raise ValueError(f"{name}: identity failed with residual {residual.to_sparse_json()}")
        checked.append(str(name))
    return {"accepted": True, "checked_identities": checked}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    with args.certificate.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        result = verify(data)
    except ValueError as exc:
        print(f"SOS identity rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
