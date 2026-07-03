import json
import unittest
from pathlib import Path

import verify_carter_sos_identity as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterSosIdentityTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_toy_identity(self):
        result = verifier.verify(self.load("certificates/carter_identities/toy_sos_identity.json"))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["checked_identities"], ["two_square_decomposition"])

    def test_rejects_bad_identity(self):
        with self.assertRaisesRegex(ValueError, "residual"):
            verifier.verify(self.load("certificates/carter_identities/bad_sos_identity.json"))

    def test_sparse_polynomial_combines_like_terms(self):
        variables = ("x", "ell")
        poly = verifier.parse_sparse_poly({"x": "1", "x^1": "2", "ell": "0"}, variables)
        self.assertEqual(poly.to_sparse_json(), {"x": "3"})

    def test_unknown_variable_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown variable"):
            verifier.eval_expr({"var": "missing"}, ("x",))


if __name__ == "__main__":
    unittest.main()
