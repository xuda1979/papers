import re
import os

def find_duplicate_labels(root_dir):
    labels = {}
    label_pattern = re.compile(r'\\label\{([^}]+)\}')
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".tex"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            matches = label_pattern.findall(line)
                            for match in matches:
                                if match in labels:
                                    labels[match].append((file_path, i + 1))
                                else:
                                    labels[match] = [(file_path, i + 1)]
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    duplicates = {k: v for k, v in labels.items() if len(v) > 1}
    return duplicates

root_dir = r"c:\Users\Lenovo\papers\yang\yang_mills\split"
duplicates = find_duplicate_labels(root_dir)

print(f"Found {len(duplicates)} duplicate labels.")
for label, locs in duplicates.items():
    print(f"\nLabel: {label}")
    for loc in locs:
        print(f"  - {loc[0]}:{loc[1]}")
