# Ascend 910B ai2 repeatability summary

Repeated-run summary from `results/ai2_torch_npu_microbench.csv`.

## Best dense vs best ephemeral by shape

- bt=1, d=1024, m=4096: dense=0.022019±0.002879 ms, best_ephemeral=0.032842±0.001082 ms at r=128, speedup(dense/eph)=0.670
- bt=1, d=2048, m=8192: dense=0.028605±0.002849 ms, best_ephemeral=0.033350±0.001465 ms at r=32, speedup(dense/eph)=0.858
- bt=1, d=4096, m=16384: dense=0.083809±0.010662 ms, best_ephemeral=0.032730±0.001293 ms at r=32, speedup(dense/eph)=2.561
- bt=8, d=1024, m=4096: dense=0.019940±0.001228 ms, best_ephemeral=0.032755±0.001377 ms at r=64, speedup(dense/eph)=0.609
- bt=8, d=2048, m=8192: dense=0.033054±0.006599 ms, best_ephemeral=0.032894±0.000664 ms at r=128, speedup(dense/eph)=1.005
- bt=8, d=4096, m=16384: dense=0.076271±0.004555 ms, best_ephemeral=0.033740±0.001110 ms at r=64, speedup(dense/eph)=2.261
- bt=32, d=1024, m=4096: dense=0.020544±0.001907 ms, best_ephemeral=0.032463±0.001369 ms at r=64, speedup(dense/eph)=0.633
- bt=32, d=2048, m=8192: dense=0.027693±0.000707 ms, best_ephemeral=0.032205±0.001149 ms at r=32, speedup(dense/eph)=0.860
- bt=32, d=4096, m=16384: dense=0.077038±0.005925 ms, best_ephemeral=0.032799±0.001661 ms at r=16, speedup(dense/eph)=2.349

## Interpretation

- Best-ephemeral wins in 4 of 9 tested shape buckets.
- Best-ephemeral losses remain in 5 of 9 tested shape buckets.
- Strongest measured win: bt=1, d=4096, m=16384, r=32, speedup=2.561.
