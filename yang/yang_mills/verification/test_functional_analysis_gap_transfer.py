import math

try:
    from verification.functional_analysis_gap_transfer import (
        transfer_gap_via_uniform_semigroup_limit,
        audit_gap_transfer_lemma_available,
    )
except Exception:
    from functional_analysis_gap_transfer import (
        transfer_gap_via_uniform_semigroup_limit,
        audit_gap_transfer_lemma_available,
    )


def test_gap_transfer_works_when_q_lt_1():
    # Choose parameters so q = delta + exp(-m t0) < 1.
    m = 2.0
    t0 = 1.0
    delta = 0.05

    res = transfer_gap_via_uniform_semigroup_limit(
        m_approx=m,
        t0=t0,
        sup_op_diff_at_t0=delta,
    )
    assert res.ok
    assert res.lower_bound > 0.0

    q = delta + math.exp(-m * t0)
    expected = -math.log(q) / t0
    assert abs(res.lower_bound - min(expected, m)) < 1e-12


def test_gap_transfer_fails_when_q_ge_1():
    m = 0.1
    t0 = 1.0
    delta = 0.95
    # q approx 0.95 + exp(-0.1) ~ 1.854... >= 1
    res = transfer_gap_via_uniform_semigroup_limit(
        m_approx=m,
        t0=t0,
        sup_op_diff_at_t0=delta,
    )
    assert not res.ok
    assert res.lower_bound == 0.0


def test_gap_transfer_audit_record_is_pass():
    chk = audit_gap_transfer_lemma_available()
    assert chk["status"] == "PASS"
    assert chk["key"] == "gap_transfer_lemma_implemented"
