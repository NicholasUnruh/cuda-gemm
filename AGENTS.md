# CUDA GEMM

- Goal: a portfolio project for GPU architecture internships. The owner writes each kernel; agents review, explain, profile and fix the harness. Ask before writing a kernel step.
- Layout: `src/gemm.cuh` holds the launcher signature, one `src/<step>.cu` per kernel, and each kernel is registered in `kKernels` in `src/bench.cu`.
- Run `make check` before committing: it compiles for sm_75, sm_80 and sm_89. No GPU here; local nvcc comes from `make toolchain` (`~/.local/opt/cuda-12.9`).
- A step counts only with a measured row from `make bench` on a real GPU: GFLOPS, % cuBLAS and `max_rel_err` within tolerance. Say which GPU. Never estimate a number.
- Matrices are row-major. cuBLAS is the reference for both speed and correctness.
- Keep the README at or below 200 words and each `docs/` file at or below 500.
