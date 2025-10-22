import math
import pytest

from pcai_reference_impl import (
    triangular_spec,
    sum_first_n_candidate,
    verify_forall_range,
    build_sum_first_n,
    wrapped_safe_divide,
    VerificationError,
)


def test_forall_range_ok_small():
    ok, cert = verify_forall_range(sum_first_n_candidate, triangular_spec, 100)
    assert ok, cert.payload
    assert cert.kind == "forall_range"
    assert "digest" not in cert.payload  # digest comes from the object, not payload


def test_build_sum_first_n_default():
    fn, cert = build_sum_first_n()
    assert cert.kind == "forall_range"
    # spot check a few values
    for n in [0, 1, 2, 10, 123]:
        assert fn(n) == triangular_spec(n)


def test_divide_ok():
    out, cert = wrapped_safe_divide(7.0, 3.0)
    assert cert.kind == "contract_trace"
    assert math.isfinite(out)


def test_divide_precondition_violation():
    with pytest.raises(VerificationError) as excinfo:
        wrapped_safe_divide(1.0, 0.0)
    cert = excinfo.value.certificate
    assert cert.kind == "contract_trace"
    assert cert.payload["ok"] is False


def test_verify_forall_range_counterexample():
    ok, cert = verify_forall_range(sum_first_n_candidate, lambda n: n, 5)
    assert not ok
    assert cert.payload["counterexample"]["n"] == 2
