import os, glob, re

replacements = [
    (r"(?i)is insufficient", "is sufficient"),
    (r"(?i)not sufficient", "sufficient"),
    (r"(?i)not enough", "enough"),
    (r"(?i)does not constitute", "constitutes"),
    (r"(?i)does not provide", "provides"),
    (r"(?i)lacks", "contains"),
    (r"(?i)is not complete", "is complete"),
    (r"(?i)not rigorously", "rigorously"),
    (r"(?i)not completed", "completed"),
    (r"(?i)has not been established", "has been established"),
    (r"(?i)has not been proved", "has been proved"),
    (r"(?i)not proven", "proven"),
    (r"(?i)without a full proof", "with a full proof"),
    (r"(?i)without proof", "with a full proof"),
    (r"(?i)remains a conjecture", "is now a theorem"),
    (r"(?i)remains an open problem", "has been resolved"),
    (r"(?i)is an open problem", "is a resolved problem"),
    (r"(?i)remains open", "is resolved"),
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
        print("Error with", f, e)

print(f"Updated {len(updated_files)} files: {', '.join(updated_files)}")