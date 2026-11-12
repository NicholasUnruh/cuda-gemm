# Backlog

## In flight
## Queued
- [ ] roofline - Roofline plot of every step on one GPU, from Nsight Compute counters (repo: cuda-gemm) (queued 2026-11-11) (hold: RTX 4060 driver returns ERR_NVGPUCTRPERM; needs GPU counters enabled for all users in the NVIDIA Control Panel on the PC; then `make profile` and `scripts/roofline.py`, both untested)
## Done
- [x] scaffold - Harness timing a kernel against cuBLAS with an error check; naive baseline; compile gate (repo: cuda-gemm) (done 2026-11-11)
- [x] first_gpu_run - `naive` measured on an RTX 4060 instead of a Colab T4 (repo: cuda-gemm) (done 2026-11-11)
- [x] coalesced - Step 2: map threadIdx.x to columns so warps read B and write C contiguously (repo: cuda-gemm) (done 2026-11-11)
- [x] smem_tiling - Step 3: shared-memory tiles of A and B (repo: cuda-gemm) (done 2026-11-11)
- [x] reg2d - Step 4: each thread computes a 2D block of C from registers (repo: cuda-gemm) (done 2026-11-11)
- [x] vec_loads - Step 5: float4 global loads, transposed A tile in shared memory (repo: cuda-gemm) (done 2026-11-11)
- [x] wmma - Step 6: tensor cores via WMMA with FP16 inputs and FP32 accumulation; FP16-input cuBLAS reference and its own tolerance (repo: cuda-gemm) (done 2026-11-11)
