# Step 2: coalesced

Source: [src/coalesced.cu](../../src/coalesced.cu). RTX 4060, 4096³: 864.3 GFLOPS, 12.4% of cuBLAS (naive: 123.6 GFLOPS).

## What changed

Only the mapping from threads to outputs. In `naive`, `threadIdx.x` picks the row, so the 32 threads of a warp compute 32 different rows of one column of C. In `coalesced`, `threadIdx.x` picks the column, so a warp computes 32 neighbouring columns of one row.

## Why it is faster

A warp issues each load as one instruction for all 32 threads, and the memory system serves it in 32-byte sectors.

| Load per k | naive warp | coalesced warp |
|---|---|---|
| `A[row*K + k]` | 32 rows, 32 sectors (stride 16 KB) | 1 address, broadcast |
| `B[k*N + col]` | 1 address, broadcast | 32 floats in a row: 4 sectors |
| store `C` | 32 sectors | 4 sectors |

For A, the naive warp touches 32 sectors (1 KB) to use 128 bytes; the rest of each sector pays off only if it is still cached when the next k needs it. The coalesced warp's B load uses every byte of the 4 sectors it touches.

Both kernels compile to the same instruction mix (59 `LDG.E`, 30 `FFMA` in the unrolled loop); only the addresses differ. The 7.0x speedup is all memory system.

## What limits it now

Every `FFMA` still needs two global loads, and threads share data only through the caches, so the load path, not arithmetic, sets the pace. Step 3 stages tiles in shared memory so each loaded float feeds 32 FMAs.

## Profile

Nsight Compute counters (sectors per request, L1 hit rate) are pending: the PC's driver blocks them (`ERR_NVGPUCTRPERM`). Counts above are from `cuobjdump -sass` and `ptxas -v` (CUDA 13.3, sm_89).
