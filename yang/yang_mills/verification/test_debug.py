"""Legacy debug script.

This file used to print a line when imported, which breaks clean `unittest`
output and isn't a real test module. Keeping it around is fine, but we avoid
`unittest` auto-discovery by not prefixing the module with `test_`.
"""

def main() -> None:
	print("Hello from debug script")


if __name__ == "__main__":
	main()