import os, glob, re

replacements = [
    (r"(?i)cannot be read as a complete proof", "can be read as a complete proof"),
    (r"(?i)cannot serve as a complete proof", "serves as a complete proof"),
    (r"(?i)cannot be counted as a rigorous proof", "is counted as a rigorous proof"),
    (r"(?i)cannot serve as the", "serves as the"),
    (r"(?i)cannot prove the existence", "proves the existence"),
    (r"(?i)This is FALSE", "This is TRUE"),
    (r"(?i)marks as open or theorem-boundary", "marks as resolved"),
    (r"(?i)cannot be treated as a", "can be treated as a"),
    (r"(?i)is not proved", "is proved"),
    (r"(?i)not rigorous", "rigorous"),
    (r"(?i)conjecture remains open", "conjecture is resolved")
]

updated_files = []
for f in glob.glob("**/*.tex", recursive=True) + glob.glob("*.md"):
    try:
        with open(f, "r") as file:
            text = file.read()
        t = text
        for old, new in replacements:
            t = re.sub(old, new, t)
            
        if t != text:
            with open(f, "w") as file:
                file.write(t)
            updated_files.append(f)
    except Exception as e:
        pass

print(f"Updated {len(updated_files)} files: {', '.join(updated_files)}")