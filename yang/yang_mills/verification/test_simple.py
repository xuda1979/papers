import sys
import unittest


class TestSimpleEnvironment(unittest.TestCase):
    def test_python_version_available(self):
        self.assertTrue(len(sys.version) > 0)

    def test_numpy_import(self):
        import numpy  # noqa: F401


if __name__ == "__main__":
    unittest.main()
