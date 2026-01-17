import re

def find_missing_labels(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all defined labels
    labels = set(re.findall(r'\\label\{([^}]+)\}', content))
    
    # Find all references
    refs = set(re.findall(r'\\ref\{([^}]+)\}', content))
    cite_refs = set(re.findall(r'\\cite\{([^}]+)\}', content)) # BibTex keys

    missing_refs = []
    for r in refs:
        if r not in labels:
            missing_refs.append(r)
            
    return missing_refs

filepath = "main_arxiv_submission.tex"
missing = find_missing_labels(filepath)

print(f"Found {len(missing)} missing references:")
for m in sorted(missing):
    print(m)
