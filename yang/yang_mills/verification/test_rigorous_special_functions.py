import os
import sys
import unittest

from mpmath import iv, mp

# Ensure imports resolve relative to verification/ directory when pytest is run
# from the repository root.
sys.path.insert(0, os.path.dirname(__file__))

from rigorous_special_functions import rigorous_besseli


class TestRigorousBesseli(unittest.TestCase):
    def test_contains_reference_point_value(self):
        # Use a modest precision reference; the enclosure should still contain it.
        mp.dps = 80
        x = 1.0
        for n in [0, 1, 2, 3]:
            enclosure = rigorous_besseli(n, iv.mpf([x, x]))
            ref = mp.besseli(n, x)
            self.assertLessEqual(enclosure.a, ref)
            self.assertGreaterEqual(enclosure.b, ref)

    def test_monotone_in_x_for_n0(self):
        # For x>=0, I_0 is increasing; interval image over [a,b] should be contained in [I0(a), I0(b)]
        mp.dps = 80
        a, b = 0.5, 2.0
        enc = rigorous_besseli(0, iv.mpf([a, b]))
        lo = mp.besseli(0, a)
        hi = mp.besseli(0, b)
        self.assertLessEqual(enc.a, lo)
        self.assertGreaterEqual(enc.b, hi)

    def test_rejects_negative_x(self):
        with self.assertRaises(ValueError):
            rigorous_besseli(0, iv.mpf([-1.0, 0.5]))


if __name__ == "__main__":
    unittest.main()
