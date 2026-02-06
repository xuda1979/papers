import math

try:
    from .interval_arithmetic import Interval
    from .interval_linear_algebra import (
        gershgorin_discs,
        gershgorin_eigenvalue_enclosure,
        interval_abs_upper,
        matrix_infinity_norm_upper,
        matrix_one_norm_upper,
    )
except ImportError:
    from interval_arithmetic import Interval
    from interval_linear_algebra import (
    gershgorin_discs,
    gershgorin_eigenvalue_enclosure,
    interval_abs_upper,
    matrix_infinity_norm_upper,
    matrix_one_norm_upper,
    )


def test_interval_abs_upper_bounds():
    assert interval_abs_upper(Interval(2.0, 3.0)) == 3.0
    assert interval_abs_upper(Interval(-5.0, -2.0)) == 5.0
    assert interval_abs_upper(Interval(-1.0, 4.0)) == 4.0


def test_matrix_norm_upper_point_matrix():
    A = [[1.0, -2.0], [3.0, 4.0]]
    # Infinity norm: max row sum of abs
    assert matrix_infinity_norm_upper(A) == 7.0
    # 1 norm: max col sum of abs
    assert matrix_one_norm_upper(A) == 6.0


def test_gershgorin_enclosure_contains_true_eigs_for_2x2():
    # Symmetric 2x2 with known eigenvalues: [[2,1],[1,2]] -> eigs {1,3}
    A = [[Interval(2.0, 2.0), Interval(1.0, 1.0)], [Interval(1.0, 1.0), Interval(2.0, 2.0)]]
    enclosure = gershgorin_eigenvalue_enclosure(A)
    assert enclosure.lower <= 1.0 <= enclosure.upper
    assert enclosure.lower <= 3.0 <= enclosure.upper

    discs = gershgorin_discs(A)
    assert len(discs) == 2
    # Each disc: center 2, radius 1
    assert discs[0][0].lower == 2.0 and discs[0][1] == 1.0
    assert discs[1][0].upper == 2.0 and discs[1][1] == 1.0
