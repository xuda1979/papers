import math
import pytest

from pcai_reference_impl import (
    build_sum_first_n,
    triangular_spec,
    sum_first_n_candidate,
    verify_forall_range,
    wrapped_safe_divide,
    VerificationError,
)


def test_forall_range_small():
    # Basic sanity: the candidate matches the spec on a small range
    ok, cert = verify_forall_range(sum_first_n_candidate, triangular_spec, 200)
    assert ok is True
    assert cert.kind == "forall_range"
    assert cert.payload["ok"] is True
    # Digest should be stable across calls (excludes timestamp)
    assert cert.digest() == cert.digest()


def test_build_sum_first_n():
    fn, cert = build_sum_first_n(500)
    assert fn(10) == triangular_spec(10)
    assert cert.payload["ok"] is True


@pytest.mark.parametrize("a,b", [(7.0, 3.0), (1.0, -2.5), (0.0, 1.0)])
def test_divide_ok(a, b):
    out, cert = wrapped_safe_divide(a, b)
    assert math.isclose(out, a / b, rel_tol=1e-9, abs_tol=1e-12)
    assert cert.kind == "contract_trace"
    assert cert.payload["ok"] is True


def test_divide_zero_rejected():
    with pytest.raises(VerificationError):
        wrapped_safe_divide(1.0, 0.0)
