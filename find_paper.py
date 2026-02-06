import os

root = r"c:\Users\david\papers"
folders_to_check = [
    "black_hole",
    "quantum_blackhole_simulation",
    "black_entangle", 
    "penrose",
    "bkl_conjecture"
]

for folder in folders_to_check:
    path = os.path.join(root, folder)
    if os.path.exists(path):
        print(f"\n--- {folder} ---")
        try:
            for f in os.listdir(path):
                print(f)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"\n--- {folder} (Not Found) ---")
