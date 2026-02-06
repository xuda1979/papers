"""
Syntax checker for all python files in verification directory
"""
import ast
import sys
import glob
import os

verification_dir = os.path.dirname(os.path.abspath(__file__))
python_files = glob.glob(os.path.join(verification_dir, "**", "*.py"), recursive=True)

errors_found = False

for filepath in python_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Try to parse the file
        ast.parse(code)
        # print(f"✓ {filepath}")
        
    except SyntaxError as e:
        print(f"✗ Syntax Error in {filepath}:")
        print(f"  Line {e.lineno}: {e.msg}")
        if e.text:
            print(f"  {e.text}")
            if e.offset:
                print(f"  {' ' * (e.offset - 1)}^")
        errors_found = True
        
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        errors_found = True
        
    except Exception as e:
        print(f"✗ Error checking {filepath}: {e}")
        errors_found = True

if not errors_found:
    print("All files passed syntax check!")
else:
    sys.exit(1)
    sys.exit(1)

print("\nSyntax check passed successfully!")
