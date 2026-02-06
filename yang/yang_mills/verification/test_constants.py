import os
import sys
import unittest


class TestConstantsModule(unittest.TestCase):
    def test_import_rigorous_constants_derivation(self):
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import rigorous_constants_derivation  # noqa: F401


if __name__ == "__main__":
    unittest.main()
