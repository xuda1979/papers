#!/usr/bin/env python3
from simulation import g2_correlation, write_table
write_table("datatableGtwo.dat", "du g2", g2_correlation(64, tau_mem=8.0))
print("Wrote datatableGtwo.dat")
