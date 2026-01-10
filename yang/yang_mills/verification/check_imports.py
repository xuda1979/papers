"""Simple import test"""
import sys
print("Python version:", sys.version)

try:
    import numpy as np
    print("✓ NumPy imported successfully, version:", np.__version__)
except ImportError as e:
    print("✗ NumPy import failed:", e)

try:
    from dataclasses import dataclass
    print("✓ dataclasses imported successfully")
except ImportError as e:
    print("✗ dataclasses import failed:", e)

try:
    from typing import List, Tuple
    print("✓ typing imported successfully")
except ImportError as e:
    print("✗ typing import failed:", e)

try:
    import json
    print("✓ json imported successfully")
except ImportError as e:
    print("✗ json import failed:", e)

try:
    from datetime import datetime
    print("✓ datetime imported successfully")
except ImportError as e:
    print("✗ datetime import failed:", e)

print("\nAll core dependencies check complete!")
