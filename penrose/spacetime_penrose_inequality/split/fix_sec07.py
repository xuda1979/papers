import os

with open("sec_07_the_p_harmonic_level_set_method.tex", "r") as f:
    text = f.read()

# Fix the min(tau, 1) lines which might have lost their backslashes
bs = chr(92)

# It seems that min(tau, 1) was parsed as min( au, 1) if backslashes were dropped somewhere.
# Wait, grep found: 
# 233:    &= O(r^{-\min(	au,1)}) = O(r^{-	au'}).
# The compilation error was:
# l.233     &
#            = O(r^{-\min(	au,1)}) = O(r^{-	au'}).
# ! Extra }, or forgotten $.
# Let's inspect lines 231-235 carefully
