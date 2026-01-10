"""
Syntax checker for tube_verifier_phase1.py
"""
import ast
import sys

filepath = r'C:\Users\Lenovo\papers\yang\yang_mills\verification\tube_verifier_phase1.py'

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Try to parse the file
    ast.parse(code)
    print(f"✓ {filepath}")
    print("  No syntax errors found!")
    
except SyntaxError as e:
    print(f"✗ Syntax Error in {filepath}:")
    print(f"  Line {e.lineno}: {e.msg}")
    print(f"  {e.text}")
    print(f"  {' ' * (e.offset - 1)}^")
    sys.exit(1)
    
except FileNotFoundError:
    print(f"✗ File not found: {filepath}")
    sys.exit(1)
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\nSyntax check passed successfully!")
