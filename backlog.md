# Backlog

## In flight
## Queued
- [ ] first_gpu_run - Run `naive` on a Colab T4; record GPU, GFLOPS and % cuBLAS in the README (repo: cuda-gemm) (queued 2026-11-11)
- [ ] coalesced - Step 2: map threadIdx.x to columns so warps read B and write C contiguously (repo: cuda-gemm) (queued 2026-11-11)
- [ ] smem_tiling - Step 3: shared-memory tiles of A and B (repo: cuda-gemm) (queued 2026-11-11)
- [ ] reg2d - Step 4: each thread computes a 2D block of C from registers (repo: cuda-gemm) (queued 2026-11-11)
- [ ] vec_loads - Step 5: float4 global loads, transposed A tile in shared memory (repo: cuda-gemm) (queued 2026-11-11)
- [ ] wmma - Step 6: tensor cores via WMMA with FP16 inputs and FP32 accumulation; FP16-input cuBLAS reference and its own tolerance (repo: cuda-gemm) (queued 2026-11-11)
- [ ] roofline - Roofline plot of every step on one GPU, from Nsight Compute counters (repo: cuda-gemm) (queued 2026-11-11)
## Done
- [x] scaffold - Harness timing a kernel against cuBLAS with an error check; naive baseline; compile gate (repo: cuda-gemm) (done 2026-11-11)
