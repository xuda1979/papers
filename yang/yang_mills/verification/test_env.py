import os
import sys
import unittest


class TestEnvironment(unittest.TestCase):
    def test_numpy_import(self):
        import numpy  # noqa: F401

    def test_rigorous_constants_import(self):
        # Ensure local package/modules resolve even when invoked from repo root.
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import rigorous_constants_derivation  # noqa: F401


if __name__ == "__main__":
    unittest.main()
