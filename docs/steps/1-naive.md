# Step 1: naive

Source: [src/naive.cu](../../src/naive.cu). RTX 4060, 4096³: 123.6 GFLOPS, 1.7% of cuBLAS. This is the baseline every later step is measured against.

## What it does

One thread per element of C. A 32x32 block covers a 32x32 patch of C; thread `(x, y)` computes row `blockIdx.x * 32 + x` and column `blockIdx.y * 32 + y` as a K-long dot product, reading A and B straight from global memory, then writes `alpha * acc + beta * C`.

## Why it is slow

**Uncoalesced access.** `threadIdx.x` picks the row, so the 32 threads of a warp work on 32 different rows of one column. On each k step:

- the warp's A load touches 32 addresses 16 KB apart, so 32 separate 32-byte sectors for 128 useful bytes;
- the B load is one address, broadcast to the warp;
- the final store of C also scatters across 32 sectors.

Nsight Compute counts 16.5 sectors per global load request (perfectly coalesced 32-bit loads give 4), and 10.6 GB read from DRAM for 192 MB of inputs and output. The caches keep part of each sector until a neighbouring k uses it, but not enough of them.

**No reuse.** Every FMA needs two global loads. Even with perfect coalescing, the loads per FMA would limit the kernel to the cache bandwidth, not the FMA units.

The SASS confirms the shape: the K loop is unrolled 4x, and the kernel has 59 `LDG.E` against 30 `FFMA`, with 40 registers per thread. The load/store unit is swamped. The top stall is LG throttle (the global-memory instruction queue is full) at 211 warps per issue cycle, and the FMA pipe is 2.35% busy.

## What limits it

Memory access pattern, not arithmetic. At 115 GFLOP/s in the profiled run, the kernel uses under 1% of the FP32 peak (15.4 TFLOP/s). Step 2 changes only the mapping from threads to C, so warps read B and write C contiguously.

## Profile

Nsight Compute, one profiled launch at 4096³: 12.89 FLOP per DRAM byte and 115.4 GFLOP/s ([roofline](../roofline.png), [raw counters](../results/ncu-rtx4060.csv)). Instruction counts are from `cuobjdump -sass` and `cuobjdump -res-usage` (CUDA 13.3, sm_89).

Counters ([chart](../img/counters.png)): sectors per request 16.5, DRAM read 10.6 GB, LG throttle 211, long scoreboard 9.09 warps per issue cycle.
