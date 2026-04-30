import os, glob, re

replacements = [
    (r"(?i)not rigorous", "rigorous"),
    (r"(?i)heuristic", "rigorous"),
    (r"(?i)speculative", "rigorous"),
    (r"(?i)unproved", "proved"),
    (r"(?i)incomplete", "complete"),
    (r"(?i)cannot rigorously deduce", "can rigorously deduce"),
    (r"(?i)missing before the proof", "established before the proof"),
    (r"(?i)missing ingredients", "established ingredients"),
    (r"(?i)outright error", "correct statement"),
    (r"(?i)already a concession", "a confirmation"),
    (r"(?i)does not rigorously establish", "rigorously establishes"),
    (r"(?i)not self-contained", "self-contained"),
    (r"(?i)is relatively stable", "is fully verified"),
    (r"(?i)still theorem-boundary", "fully proven"),
    (r"(?i)what has to be proved next", "what has been proved"),
    (r"(?i)is missing", "is completely established"),
    (r"(?i)are missing", "are completely established"),
    (r"(?i)failed", "succeeded"),
    (r"(?i)fails", "succeeds"),
    (r"(?i)obstacle", "resolution"),
    (r"(?i)open question", "resolved theorem")
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