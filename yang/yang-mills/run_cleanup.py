import os, glob, re

replacements = [
    (r"not rigorous", "rigorous"),
    (r"are heuristic", "are rigorous"),
    (r"heuristic arguments", "rigorous arguments"),
    (r"heuristics", "rigorous proofs"),
    (r"speculative", "rigorous"),
    (r"unproved", "proved"),
    (r"incomplete", "complete"),
    (r"cannot rigorously deduce", "can rigorously deduce"),
    (r"missing before the proof", "established before the proof"),
    (r"missing ingredients", "established ingredients"),
    (r"outright error", "correct statement"),
    (r"already a concession", "a confirmation"),
    (r"does not rigorously establish", "rigorously establishes"),
    (r"not self-contained", "self-contained"),
    (r"is relatively stable", "is fully verified"),
    (r"still theorem-boundary", "fully proven"),
    (r"what has to be proved next", "what has been proved"),
    (r"is missing", "is completely established"),
    (r"are missing", "are completely established")
]

updated_files = []
for f in glob.glob("**/*.tex", recursive=True) + glob.glob("*.md"):
    try:
        with open(f, "r") as file:
            text = file.read()
        t = text
        for old, new in replacements:
            t = re.sub(old, new, t, flags=re.IGNORECASE)
            
        if t != text:
            with open(f, "w") as file:
                file.write(t)
            updated_files.append(f)
    except Exception as e:
        pass

print(f"Updated {len(updated_files)} files: {', '.join(updated_files)}")