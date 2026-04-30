# Ascend 910B ai2 application-scenario summary

Repeated scenario benchmark comparing dense, static ephemeral, and dynamic ephemeral variants.

- Static ephemeral beats dense in 22/27 scenario-shape buckets.
- Dynamic ephemeral beats dense in 9/27 scenario-shape buckets.

## Best static and dynamic speedups by shape

- scenario=ffn, d=2048, bt=1: best_static r=64, speedup=1.522; best_dynamic r=32, speedup=0.374
- scenario=ffn, d=2048, bt=8: best_static r=32, speedup=0.596; best_dynamic r=32, speedup=0.191
- scenario=ffn, d=2048, bt=32: best_static r=64, speedup=0.976; best_dynamic r=16, speedup=0.270
- scenario=ffn, d=4096, bt=1: best_static r=32, speedup=3.799; best_dynamic r=32, speedup=0.636
- scenario=ffn, d=4096, bt=8: best_static r=64, speedup=3.782; best_dynamic r=16, speedup=0.682
- scenario=ffn, d=4096, bt=32: best_static r=16, speedup=3.643; best_dynamic r=16, speedup=0.787
- scenario=ffn, d=8192, bt=1: best_static r=64, speedup=15.466; best_dynamic r=16, speedup=3.650
- scenario=ffn, d=8192, bt=8: best_static r=32, speedup=16.594; best_dynamic r=32, speedup=3.113
- scenario=ffn, d=8192, bt=32: best_static r=64, speedup=14.335; best_dynamic r=16, speedup=3.844
- scenario=moe_top2, d=2048, bt=1: best_static r=64, speedup=1.151; best_dynamic r=32, speedup=0.486
- scenario=moe_top2, d=2048, bt=8: best_static r=32, speedup=0.889; best_dynamic r=32, speedup=0.392
- scenario=moe_top2, d=2048, bt=32: best_static r=16, speedup=0.834; best_dynamic r=32, speedup=0.390
- scenario=moe_top2, d=4096, bt=1: best_static r=32, speedup=2.303; best_dynamic r=32, speedup=0.826
- scenario=moe_top2, d=4096, bt=8: best_static r=32, speedup=2.866; best_dynamic r=16, speedup=0.957
- scenario=moe_top2, d=4096, bt=32: best_static r=16, speedup=2.402; best_dynamic r=16, speedup=0.860
- scenario=moe_top2, d=8192, bt=1: best_static r=64, speedup=7.378; best_dynamic r=16, speedup=3.530
- scenario=moe_top2, d=8192, bt=8: best_static r=32, speedup=8.637; best_dynamic r=16, speedup=3.410
- scenario=moe_top2, d=8192, bt=32: best_static r=64, speedup=9.224; best_dynamic r=16, speedup=3.230
- scenario=swiglu, d=2048, bt=1: best_static r=64, speedup=1.058; best_dynamic r=64, speedup=0.282
- scenario=swiglu, d=2048, bt=8: best_static r=64, speedup=2.051; best_dynamic r=32, speedup=0.647
- scenario=swiglu, d=2048, bt=32: best_static r=32, speedup=0.979; best_dynamic r=32, speedup=0.265
- scenario=swiglu, d=4096, bt=1: best_static r=16, speedup=3.105; best_dynamic r=16, speedup=0.716
- scenario=swiglu, d=4096, bt=8: best_static r=16, speedup=3.841; best_dynamic r=32, speedup=0.794
- scenario=swiglu, d=4096, bt=32: best_static r=16, speedup=3.678; best_dynamic r=16, speedup=0.769
- scenario=swiglu, d=8192, bt=1: best_static r=64, speedup=11.149; best_dynamic r=16, speedup=3.030
- scenario=swiglu, d=8192, bt=8: best_static r=64, speedup=10.453; best_dynamic r=32, speedup=3.011
- scenario=swiglu, d=8192, bt=32: best_static r=16, speedup=15.783; best_dynamic r=16, speedup=3.644
