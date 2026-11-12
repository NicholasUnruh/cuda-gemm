# Step 5: vectorized loads, transposed A tile

Source: [src/vec.cu](../../src/vec.cu). RTX 4060, 4096³: 7,053 GFLOPS, 88.1% of cuBLAS (reg2d: 6,545).

## What changed

Step 4's tiling with 128-bit memory operations:

- Each thread loads one `float4` of A and one of B per K step, instead of four scalar loads of each.
- A is stored transposed in shared memory (`As[k][m]`), so a thread's 8 values of A for one k are contiguous and come back as two `float4` reads, like B.
- C moves as `float4` too.

## Why it is faster

Same bytes, a quarter of the instructions: per K step the SASS has 2 `LDG.E.128` where step 4 had 8 `LDG.E`; the C epilogue has 16 `LDG.E.128` and 16 `STG.E.128` where step 4 had 64 of each. Fewer memory instructions leave more issue slots for FMAs.

## What the transposed tile does

In step 4 the compiler had already turned the A reads into `LDS.128`: with K step 8, a thread's A values for all eight k are contiguous once the loop is unrolled. Both kernels issue 32 `LDS.128` per K step. Here the transpose makes them vectorizable by construction, at a cost: the transposed store is four scalar `STS`, and the two threads that load one row of A write the same bank (rows 512 words apart), a 2-way bank conflict on that store. The gain over step 4 comes from the global side.

## What limits it now

Nothing overlaps the next tile's global load with this tile's math (double buffering would), and FP32 FMAs cap throughput. Step 6 moves the math to tensor cores.

## Profile

Nsight Compute, one profiled launch at 4096³: 58.26 FLOP per DRAM byte and 8352.6 GFLOP/s ([roofline](../roofline.png), [raw counters](../results/ncu-rtx4060.csv)). Instruction counts above are from `cuobjdump -sass` and `ptxas -v` (CUDA 13.3, sm_89).
