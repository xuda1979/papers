import re

file_path = r'yang_mills/split/app204_mathematical_audit_summary.tex'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    derivations = re.findall(r'\\subsubsection\{Derivation\s+([A-Z]+):\s+(.*?)\}', content)
    print(f"Found {len(derivations)} derivations")
    print("Derivation tags:", [d[0] for d in derivations])

    # Look for numbered errors in table or list
    # Pattern: \textbf{1. ...} or \textbf{55. ...}
    numbered = re.findall(r'\\textbf\{(\d+)\.\s+(.*?)\}', content)
    
    # Filter out small numbers if they are just enumeration noise, but here we expect 1-55
    nums = [int(n[0]) for n in numbered]
    print(f"Found {len(nums)} numbered items")
    print("Item numbers:", sorted(list(set(nums))))

except Exception as e:
    print(f"Error: {e}")
