# Step 3: shared-memory tiling

Source: [src/smem.cu](../../src/smem.cu). RTX 4060, 4096³: 1,187 GFLOPS, 16.3% of cuBLAS (coalesced: 864.3).

## What changed

A 32x32 block owns a 32x32 tile of C and walks K in steps of 32. Each step, every thread copies one float of A and one of B into shared-memory tiles (`As`, `Bs`, 8 KB), syncs, and each thread does 32 FMAs reading only shared memory.

## Why it is faster

In `coalesced`, every thread loads its own row of A and column of B from global memory, so each element is fetched by 32 threads of the block. With the tile the block fetches it once: global loads per FMA fall from 2 to 2/32.

The measured gain is only 1.4x, far less than the 32x cut in global load instructions: most of `coalesced`'s loads probably already hit L1/L2, so the cut mainly saved cache traffic.

The inner loop has no bank conflicts: `Bs[k][tx]` is 32 consecutive words (one per bank) and `As[ty][k]` is one word broadcast to the warp.

## What limits it now

The SASS of the inner loop has 33 `FFMA` against 32 `LDS` and 8 `LDS.128`: about one shared load per FMA. The load/store unit issues nearly as many instructions as the FMA pipe, so shared-memory traffic, not DRAM, is the likely limit.

Occupancy is fixed by the block size: 1024 threads at 38 registers (allocated as 40) is 40,960 registers, so only one block fits in the SM's 65,536, i.e. 32 of 48 warps.

Step 4 makes each shared load feed several FMAs.

## Profile

Nsight Compute counters (shared-memory throughput, bank conflicts, stall reasons) are pending: the PC's driver blocks them (`ERR_NVGPUCTRPERM`). Counts above are from `cuobjdump -sass` and `ptxas -v` (CUDA 13.3, sm_89).
