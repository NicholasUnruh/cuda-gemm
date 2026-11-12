# CUDA GEMM

Matrix multiply in CUDA, step by step from a naive kernel to tensor cores. Each step is timed and checked against cuBLAS on the same inputs.

| Step | Kernel | Idea | GPU | GFLOPS (4096³) | % cuBLAS |
|---|---|---|---|---:|---:|
| 1 | `naive` | One thread per output | RTX 4060 | 123.6 | 1.7 |
| 2 | [`coalesced`](docs/steps/2-coalesced.md) | Warps read B and write C contiguously | RTX 4060 | 864.3 | 12.4 |
| 3 | `smem` | Shared-memory tiles | — | — | — |
| 4 | `reg2d` | 2D register blocking per thread | — | — | — |
| 5 | `vec` | 128-bit loads, transposed A tile | — | — | — |
| 6 | `wmma` | Tensor cores (FP16 in, FP32 accumulate) | — | — | — |

sm_89, CUDA 13.3, median of three runs ([raw](docs/results/rtx4060.csv)).

```sh
make check               # compile every kernel; no GPU needed
make bench KERNEL=naive  # on a GPU: CSV of time, GFLOPS, % cuBLAS, error
```

Running on a GPU: [docs/running.md](docs/running.md).
