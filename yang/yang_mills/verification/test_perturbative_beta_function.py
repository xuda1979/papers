import unittest


class TestPerturbativeBetaFunction(unittest.TestCase):
    def test_beta_function_has_correct_sign(self):
        # Beta function in UV should be negative for small positive alpha
        try:
            from verification.verify_perturbative_regime import beta_function_2loop
            from mpmath import iv
        except Exception as exc:
            # If mpmath isn't installed in this environment, skip (the main script already hard-requires it).
            self.skipTest(f"mpmath/iv not available: {exc}")
            return

        alpha = iv.mpf([0.01, 0.01])
        beta = beta_function_2loop(alpha)
        # beta is an interval; upper bound should be < 0
        self.assertLess(float(beta.b), 0.0)


if __name__ == "__main__":
    unittest.main()
