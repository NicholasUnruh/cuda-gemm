# Step 4: 2D register blocking

Source: [src/reg2d.cu](../../src/reg2d.cu). RTX 4060, 4096³: 6,545 GFLOPS, 80.4% of cuBLAS (smem: 1,187).

## What changed

A block of 256 threads owns a 128x128 tile of C and walks K in steps of 8 (tiles: A 128x8, B 8x128, 8 KB). Each thread computes an 8x8 block of C kept in 64 accumulator registers. Per k it reads 8 values of A (a column of its rows) and 8 of B (a row of its columns) into registers and does the 8x8 outer product: 64 FMAs.

## Why it is faster

Reuse moves into registers: 16 loaded values feed 64 FMAs, and the 128x128 tile reuses each global float 128 times instead of 32.

The compiler goes further: after unrolling the 8-step k loop, a thread's 8 values of A for one row are contiguous in `As`, so it reads them with `LDS.128`. Per K step the SASS has 512 `FFMA` and 32 `LDS.128`: 16 FMAs per shared-load instruction, against about 1 in step 3.

## Trade-off: occupancy

64 accumulators plus operands take 128 registers per thread (no spills). 256 x 128 = 32,768 registers, so two blocks fit per SM: 16 of 48 warps. That is acceptable: each warp has 64 independent FMAs per k, and this instruction-level parallelism hides latency that more warps otherwise would.

## What limits it now

Global memory is still accessed 4 bytes per instruction: 8 `LDG.E` per K step, and 64 `LDG.E` plus 64 `STG.E` to read and write each thread's 8x8 block of C. Step 5 issues them 16 bytes at a time.

## Profile

Nsight Compute, one profiled launch at 4096³: 57.08 FLOP per DRAM byte and 7044.4 GFLOP/s ([roofline](../roofline.png), [raw counters](../results/ncu-rtx4060.csv)). Instruction counts above are from `cuobjdump -sass` and `ptxas -v` (CUDA 13.3, sm_89).

Counters against `smem` ([chart](../img/counters.png)): MIO throttle falls from 23.6 to 0.82, the FMA pipe goes from 7.8% to 48.8% busy and DRAM reads from 8.92 to 2.34 GB, while achieved occupancy halves (66.7% to 33.0%) as predicted above.
