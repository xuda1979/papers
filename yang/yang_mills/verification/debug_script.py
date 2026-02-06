"""Legacy debug helper.

This module is intentionally *not* named `test_*.py` so it won't be picked up by
`python -m unittest` discovery.
"""


def main() -> None:
    print("Hello from debug script")


if __name__ == "__main__":
    main()
