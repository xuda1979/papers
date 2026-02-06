"""Unit tests for key verification components.

These were originally quick print-based sanity checks, but are now structured
as `unittest` tests so CI / `python -m unittest` can reliably detect failures.
"""

import os
import sys
import unittest


class TestPhase1Components(unittest.TestCase):
    def setUp(self):
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

    def test_interval_arithmetic_addition(self):
        from tube_verifier_phase1 import Interval

        I1 = Interval(1.0, 2.0)
        I2 = Interval(0.5, 1.5)
        I3 = I1 + I2
        self.assertLessEqual(I3.lower, 1.5)
        self.assertGreaterEqual(I3.upper, 3.5)

    def test_operator_basis_dimensions_nonempty(self):
        from tube_verifier_phase1 import OperatorBasis

        dims = OperatorBasis.dimensions()
        self.assertTrue(isinstance(dims, (list, tuple)))
        self.assertGreater(len(dims), 0)

    def test_rg_map_initializes(self):
        from tube_verifier_phase1 import RGMap

        rg = RGMap(L=2, N=3)
        self.assertEqual(rg.L, 2)
        self.assertEqual(rg.N, 3)

    def test_tube_radius_positive(self):
        from tube_verifier_phase1 import TubeDefinition

        tube = TubeDefinition(0.3, 2.4, N=3)
        r = tube.radius(1.0)
        self.assertGreater(r, 0.0)


if __name__ == "__main__":
    unittest.main()
