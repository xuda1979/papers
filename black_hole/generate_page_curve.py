#!/usr/bin/env python3
from simulation import page_curve, write_table
write_table("datatablePagecurve.dat", "t S mean-entropy", page_curve(12,12))
print("Wrote datatablePagecurve.dat")
