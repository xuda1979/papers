#!/usr/bin/env python3
from simulation import ablation_sweep, write_table
rows = ablation_sweep(l_mem_true=6, l_range=(1,16), tau_mem=8.0)
write_table("datatableAblation.dat", "ell deficit", [(r[0], r[1]) for r in rows])
write_table("datatableAblationSig.dat", "ell sig", [(r[0], r[2]) for r in rows])
print("Wrote datatableAblation.dat, datatableAblationSig.dat")
