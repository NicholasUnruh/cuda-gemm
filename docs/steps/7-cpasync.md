# Step 7: cp.async double buffering

Source: [src/cpasync.cu](../../src/cpasync.cu). RTX 4060, 4096³: 28,823 GFLOPS, 95.3% of cuBLAS FP16 (`wmma`: 20,731, 65.3%).

## What changed

The math is step 6's: the same 128x128 block tile, 64x32 warp tiles and WMMA fragments. Two things around it changed:

1. **Pipelined K loop.** A and B each get two shared-memory buffers. `cp.async` copies 16 bytes from global to shared memory without passing through registers. While the warps multiply tile `t` from one buffer, tile `t+1` lands in the other.
2. **Grouped block order.** Block `i` maps to a tile of C column by column inside groups of 8 block rows (`kGroup`), not row by row.

One `__syncthreads` per K tile is enough. At the top of iteration `t`, each thread waits for its own copies of tile `t`. The barrier then makes all copies visible and guarantees every warp has finished tile `t-1`. Only after it are tile `t+1`'s copies issued, into tile `t-1`'s buffer.

## Why it is faster

In `wmma` each K tile goes load, barrier, compute, barrier, so the tensor cores sit out every load: long-scoreboard stalls of 15.2 warps per issue cycle, with the tensor pipe 33.5% busy. Here those stalls fall to 0.92 and the tensor pipe rises to 47.3%. The SASS shows `LDGSTS.E.BYPASS.128` (the `.cg` copy skips L1), `LDGDEPBAR` per commit and `DEPBAR.LE SB0` per wait. With no staging registers the kernel uses 122 registers and 37,888 bytes of shared memory, so two blocks still fit on each SM.

## Grouped block order

Pipelining alone reached 27,682 GFLOPS. Its counters showed DRAM reads of 1.08 GB, although A and B total 64 MB. The 48 resident blocks span about 1.5 block rows and read all 32 MB of B, which does not fit in the 24 MB L2, so every block row fetches B from DRAM again. In groups of 8 rows, the resident blocks touch about 14 MB.

| Block order | GFLOPS | % cuBLAS | DRAM read | L2 hit rate |
|---|---:|---:|---:|---:|
| Row-major (`kGroup = 1`) | 27,682 | 92.1 | 1,078 MB | 67.9% |
| Groups of 8 rows | 28,823 | 95.3 | 242 MB | 92.8% |

Raw data: [cpasync-order.csv](../results/cpasync-order.csv). Groups of 4 and 16 fall within run-to-run noise of groups of 8.

## What limits it now

The tensor pipe: math-pipe throttle is the top stall (13.6), ahead of the barrier (4.1). GeForce Ada runs FP16 MMA with FP32 accumulation at half the FP16-accumulate rate (Ada whitepaper). The roofline's 61.5 TFLOP/s ceiling is the counter's full rate, so about half of it is reachable. That fits cuBLAS at 30–32 TFLOP/s and the pipe stopping just under 50%. The card is power-limited (115 W), so cuBLAS varies 2–3 points from run to run.

## Profile

Nsight Compute, one profiled launch at 4096³: 451.49 FLOP per DRAM byte and 28,839.9 GFLOP/s ([roofline](../roofline.png), [raw counters](../results/ncu-rtx4060.csv); profiled 2026-11-12). SASS from `cuobjdump -sass` and `-res-usage` (CUDA 13.3, sm_89). Process notes: [journal](../journal/2026-11-12-cpasync.md).
