# CUDA GEMM

Single-precision matrix multiply in CUDA, one step at a time, from a naive kernel to tensor cores. Each step is timed against cuBLAS and checked for correctness on the same inputs.

| Step | Kernel | Idea | GFLOPS (4096³) | % cuBLAS |
|---|---|---|---:|---:|
| 1 | `naive` | One thread per output | — | — |
| 2 | `coalesced` | Warps read rows of A and B contiguously | — | — |
| 3 | `smem` | Shared-memory tiles | — | — |
| 4 | `reg2d` | 2D register blocking per thread | — | — |
| 5 | `vec` | 128-bit loads, transposed A tile | — | — |
| 6 | `wmma` | Tensor cores (FP16 in, FP32 accumulate) | — | — |

GPU: — (filled in with the first measured run).

```sh
make check               # compile every kernel; no GPU needed
make bench KERNEL=naive  # on a GPU: CSV of time, GFLOPS, % cuBLAS, error
```

Running on Colab or a rented GPU: [docs/running.md](docs/running.md).
