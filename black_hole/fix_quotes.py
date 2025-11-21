#!/usr/bin/env python3
"""Fix quote characters to em-dashes in paper.tex"""

# Read the file
with open('paper.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the specific quote pattern with em-dashes
# The pattern uses Unicode right single quote (U+2019) and left double quote (U+201C)
old_pattern = "deriving such dynamics\u2019\u201cespecially the randomized couplings $J_{ab}$\u2019\u201cdirectly from"
new_pattern = "deriving such dynamics---especially the randomized couplings $J_{ab}$---directly from"

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    print(f"✓ Replaced quote marks with em-dashes")
else:
    print(f"✗ Pattern not found. Looking for similar patterns...")
    # Try to find similar text
    if "deriving such dynamics" in content:
        idx = content.find("deriving such dynamics")
        print(f"Found 'deriving such dynamics' at position {idx}")
        sample = content[idx:idx+120]
        print(f"Context: {repr(sample)}")
        # Show the exact bytes
        for i, char in enumerate(sample[18:30]):
            print(f"  pos {18+i}: {repr(char)} = U+{ord(char):04X}")

# Write back
with open('paper.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")

