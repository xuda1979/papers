#!/usr/bin/env python3
import sys, subprocess
subprocess.check_call([sys.executable, "simulation.py",
    "--s-initial","12","--steps","12","--num-runs","100",
    "--tau-mem","8.0","--l-mem","6","--seed","42",
    "--threads","1","--pythonhashseed","0","--save-ledger","seed_ledger.json"])
print("Done: generate_all")

