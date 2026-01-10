import os

print("Scanning for small files...")
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.tex'):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                    if len(content.strip()) < 200:
                        print(f"SMALL FILE: {path} ({len(content.strip())} chars)")
            except Exception as e:
                print(f"Error reading {path}: {e}")
