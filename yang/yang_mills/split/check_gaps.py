import os

print("Checking files for gaps...")
for f in os.listdir('.'):
    if f.endswith('.tex'):
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                is_short = len(content) < 500
                has_todo = 'TODO' in content
                has_fixme = 'FIXME' in content
                has_gap = 'GAP' in content
                
                if is_short or has_todo or has_fixme or has_gap:
                    print(f"{f}: Size={len(content)}, TODO={has_todo}, FIXME={has_fixme}, GAP={has_gap}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
