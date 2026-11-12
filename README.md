# CUDA GEMM

Matrix multiply in CUDA, step by step from a naive kernel to tensor cores. Each step is timed and checked against cuBLAS on the same inputs.

| Step | Kernel | Idea | GPU | GFLOPS (4096³) | % cuBLAS |
|---|---|---|---|---:|---:|
| 1 | `naive` | One thread per output | RTX 4060 | 123.6 | 1.7 |
| 2 | [`coalesced`](docs/steps/2-coalesced.md) | Warps read B and write C contiguously | RTX 4060 | 864.3 | 12.4 |
| 3 | [`smem`](docs/steps/3-smem.md) | Shared-memory tiles | RTX 4060 | 1,187 | 16.3 |
| 4 | [`reg2d`](docs/steps/4-reg2d.md) | 2D register blocking per thread | RTX 4060 | 6,545 | 80.4 |
| 5 | [`vec`](docs/steps/5-vec.md) | 128-bit loads, transposed A tile | RTX 4060 | 7,053 | 88.1 |
| 6 | [`wmma`](docs/steps/6-wmma.md) | Tensor cores (FP16 in, FP32 accumulate) | RTX 4060 | 20,731 | 65.3 |

sm_89, CUDA 13.3, median of three runs ([raw](docs/results/rtx4060.csv)); `wmma` against FP16-input cuBLAS.

```sh
make check               # compile every kernel; no GPU needed
make bench KERNEL=naive  # on a GPU: CSV of time, GFLOPS, % cuBLAS, error
```

Running on a GPU: [docs/running.md](docs/running.md).
