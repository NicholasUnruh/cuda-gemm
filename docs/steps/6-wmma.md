# Step 6: WMMA tensor cores

Source: [src/wmma.cu](../../src/wmma.cu). RTX 4060, 4096³: 20,731 GFLOPS, 65.3% of cuBLAS FP16 (`cublasGemmEx`, FP16 in, FP32 accumulate).

## What changed

Inputs are FP16; accumulation and C stay FP32. A block of 8 warps owns a 128x128 tile of C and walks K in steps of 32. Each warp owns 64x32 of C as 4x2 accumulator fragments of 16x16 and, per 16 of K, loads 4 A and 2 B fragments and issues 8 `mma_sync` calls. Tiles reach shared memory through 16-byte loads.

The harness gained an FP16 path: it converts A and B to FP16 once, untimed, and checks against `cublasGemmEx` on the same FP16 inputs. Only summation order differs, so the tolerance is 1e-3 (FP32 kernels: 1e-4).

## Why it is faster

One warp-wide `mma_sync` does a 16x16x16 multiply-accumulate (8,192 FLOPs) on tensor cores; it compiles to two `HMMA.16816.F32`. The SASS shows 32 `HMMA` per K step. Fragments come from shared memory through `LDSM` (`ldmatrix`), with `.MT` transposing B on the fly.

## Bank conflicts and padding

`ldmatrix` reads 8 rows of 16 bytes each. Unpadded A rows are 64 bytes apart and B rows 256 bytes, so the 8 rows share banks. Padding each row by 8 halves (16 bytes) gives strides of 20 and 68 words; 8 consecutive rows then cover all 32 banks once. Fragment pointers stay 32-byte aligned, as `load_matrix_sync` requires.

## What limits it now

No pipelining: each K step loads, syncs, computes and syncs, so tensor cores idle during loads. Overlapping them (`cp.async` into two shared buffers) is the next step. The epilogue also reads C although beta is 0.

## Profile

Nsight Compute counters (tensor-pipe utilisation, stalls) are pending: the PC's driver blocks them (`ERR_NVGPUCTRPERM`). Counts above are from `cuobjdump -sass` and `ptxas -v` (CUDA 13.3, sm_89).
