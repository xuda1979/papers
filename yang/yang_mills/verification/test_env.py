import sys
import os
print("Hello World")
try:
    import numpy
    print("Numpy OK")
except ImportError as e:
    print(e)

path = r'c:\Users\Lenovo\papers\yang\yang_mills\verification'
if path not in sys.path:
    sys.path.append(path)

try:
    import rigorous_constants_derivation
    print("Rigorous Constants OK")
except ImportError as e:
    print(f"Rigorous Constants Fail: {e}")
