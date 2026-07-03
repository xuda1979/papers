import copy
import json
import unittest
from fractions import Fraction
from pathlib import Path

import verify_carter_gram_certificate as verifier


ROOT = Path(__file__).resolve().parents[1]


class VerifyCarterGramCertificateTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_accepts_toy_psd_gram_identity(self):
        data = self.load("certificates/carter_identities/toy_gram_certificate.json")
        result = verifier.verify(data)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["basis_size"], 3)
        self.assertEqual(result["checked_principal_minors"], 7)

    def test_rejects_indefinite_gram_matrix(self):
        data = self.load("certificates/carter_identities/bad_indefinite_gram_certificate.json")
        with self.assertRaisesRegex(ValueError, "principal minor"):
            verifier.verify(data)

    def test_rejects_bad_gram_identity(self):
        data = self.load("certificates/carter_identities/bad_gram_identity_certificate.json")
        with self.assertRaisesRegex(ValueError, "Gram identity failed"):
            verifier.verify(data)

    def test_rejects_nonsymmetric_matrix(self):
        data = copy.deepcopy(self.load("certificates/carter_identities/toy_gram_certificate.json"))
        data["gram_matrix"][0][1] = "0"
        with self.assertRaisesRegex(ValueError, "symmetric"):
            verifier.verify(data)

    def test_exact_determinant(self):
        matrix = [
            [Fraction(2), Fraction(1)],
            [Fraction(1), Fraction(2)],
        ]
        self.assertEqual(verifier.determinant(matrix), Fraction(3))


if __name__ == "__main__":
    unittest.main()
