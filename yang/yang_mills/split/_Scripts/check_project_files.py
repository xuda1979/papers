import re
import os

main_tex = r"c:\Users\Lenovo\papers\yang\yang_mills\split\yang_mills_large_rigorous.tex"
base_dir = os.path.dirname(main_tex)

with open(main_tex, 'r') as f:
    content = f.read()

includes = re.findall(r'\\input\{([^}]+)\}', content)

missing_files = []
for filename in includes:
    if not os.path.exists(os.path.join(base_dir, filename)):
        missing_files.append(filename)

print(f"Total includes: {len(includes)}")
print(f"Missing files: {len(missing_files)}")
for f in missing_files:
    print(f"MISSING: {f}")
