import json
import unittest
from pathlib import Path

import solve_carter_gram_certificate as solver
import verify_carter_gram_certificate as verifier


ROOT = Path(__file__).resolve().parents[1]


class SolveCarterGramCertificateTests(unittest.TestCase):
    def load(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_solves_toy_rank_one_plus_diagonal_certificate(self):
        data = self.load("certificates/carter_identities/toy_gram_solve_input.json")
        result = solver.solve(data)
        self.assertEqual(
            result["gram_matrix"],
            [["1", "1", "0"], ["1", "1", "0"], ["0", "0", "1"]],
        )
        self.assertEqual(len(result["solver_decomposition"]), 2)
        self.assertTrue(verifier.verify(result)["accepted"])

    def test_rejects_unsolved_non_sos_target(self):
        data = self.load("certificates/carter_identities/bad_gram_solve_input.json")
        with self.assertRaisesRegex(ValueError, "failed to extract"):
            solver.solve(data)

    def test_rejects_too_small_support_bound(self):
        data = self.load("certificates/carter_identities/toy_gram_solve_input.json")
        with self.assertRaisesRegex(ValueError, "failed to extract"):
            solver.solve(data, max_support=1)


if __name__ == "__main__":
    unittest.main()
