# Ascend 910B ai2 frontier summary

Denser five-trial rank sweep collected on 2026-03-25 for `d in {2048,4096}`, `b in {1,8,32}`, and `r in {8,16,24,32,48,64,96,128,192,256}`.

## Best rank by shape

- d=2048, bt=1: best rank=64, dense=0.029932 ms, best_ephemeral=0.032594 ms, speedup(dense/eph)=0.918
- d=2048, bt=8: best rank=32, dense=0.030896 ms, best_ephemeral=0.032331 ms, speedup(dense/eph)=0.956
- d=2048, bt=32: best rank=16, dense=0.028677 ms, best_ephemeral=0.031774 ms, speedup(dense/eph)=0.903
- d=4096, bt=1: best rank=16, dense=0.076526 ms, best_ephemeral=0.032375 ms, speedup(dense/eph)=2.364
- d=4096, bt=8: best rank=32, dense=0.081938 ms, best_ephemeral=0.031936 ms, speedup(dense/eph)=2.566
- d=4096, bt=32: best rank=16, dense=0.081343 ms, best_ephemeral=0.032614 ms, speedup(dense/eph)=2.494

## Interpretation

- The denser sweep removes the last plausible `d=2048` crossover claim in the current measurement framework: all measured `d=2048` buckets remain below parity even after a finer rank search.
- The `d=4096` win region is broad rather than fragile. Ranks in roughly the `16-64` band remain consistently strong.
- Some ranks are materially worse despite similar FLOP counts, especially `r=8` and `r=24`, which indicates kernel-shape effects beyond simple arithmetic accounting.
