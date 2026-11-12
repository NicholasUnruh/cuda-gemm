# CUDA GEMM

Matrix multiply in CUDA, built step by step from a naive kernel to tensor cores, with every step timed and checked against cuBLAS on the same inputs.

| Step | Kernel | Idea | GFLOPS (4096³) | % cuBLAS |
|---|---|---|---:|---:|
| 1 | [`naive`](docs/steps/1-naive.md) | One thread per output | 123.6 | 1.7 |
| 2 | [`coalesced`](docs/steps/2-coalesced.md) | Warps read B and write C contiguously | 864.3 | 12.4 |
| 3 | [`smem`](docs/steps/3-smem.md) | Shared-memory tiles | 1,187 | 16.3 |
| 4 | [`reg2d`](docs/steps/4-reg2d.md) | 2D register blocking per thread | 6,545 | 80.4 |
| 5 | [`vec`](docs/steps/5-vec.md) | 128-bit loads, transposed A tile | 7,053 | 88.1 |
| 6 | [`wmma`](docs/steps/6-wmma.md) | Tensor cores (FP16 in, FP32 accumulate) | 20,731 | 65.3 |
| 7 | [`cpasync`](docs/steps/7-cpasync.md) | `cp.async` double buffering, grouped block order | 28,823 | 95.3 |
| 8 | [`fp8`](docs/steps/8-fp8.md) | FP8 (E4M3) `mma.sync`, FP32 promotion per K tile | 57,778 | 93.3 |

Measured on an RTX 4060 (sm_89, CUDA 13.3), median of three runs ([raw](docs/results/rtx4060.csv)). `wmma` and `cpasync` are compared with FP16-input cuBLAS, `fp8` with cuBLASLt FP8.

![Roofline of every kernel on the RTX 4060](docs/roofline.png)

## Why it matters

Each step removes one bottleneck and explains the gain from SASS and Nsight Compute counters:

![Nsight Compute counters per kernel](docs/img/counters.png)

Tiles per step, to scale; their implied global loads match the counters for steps 1–5:

![Block, warp and thread tiles per step](docs/img/tiling.png)

## Quick start

```sh
make toolchain                    # optional: local nvcc and cuBLAS (micromamba)
make check                        # compile for sm_75, sm_80, sm_89; no GPU needed
make bench ARCH=sm_89 KERNEL=vec  # on a GPU: time, GFLOPS, % cuBLAS, error as CSV
```

## Repository layout

| Path | Contents |
|---|---|
| `src/gemm.cuh` | Launcher signatures shared by every kernel |
| `src/<step>.cu` | One kernel per step |
| `src/bench.cu` | Harness: timing, cuBLAS reference, error check |
| `src/quant.cu`, `src/numerics.cu` | Quantizers; per-format error and cost (`bench numerics`) |
| `check/` | Tensor-core adder probe; quantizer cross-check against chip-lab |
| `scripts/` | Roofline, counter and tiling plots from the result CSVs |
| `docs/steps/`, `docs/journal/` | One note per step; dated process notes |
| `docs/results/` | Raw benchmark and Nsight Compute CSVs |

## Results and verification

- Every run checks `max_rel_err` and fails above 1e-4 (FP32) or 1e-3 (FP16, FP8; FP8 against exact dot products).
- On chip-lab's test distributions, GPU FP8 and BF16 error match its bit-exact C++ models ([step 8](docs/steps/8-fp8.md)).
- Running, profiling and plots: [docs/running.md](docs/running.md).

## Limitations

- One GPU (RTX 4060 under WSL2); results on other architectures will differ.
- Tiled kernels need M, N and K to be multiples of 128; there is no edge handling.

## Related projects

- [openroad-chip-lab](https://github.com/NicholasUnruh/openroad-chip-lab): GEMM tiles and number formats in custom silicon.
- [bf16-fpga](https://github.com/NicholasUnruh/bf16-fpga): that tile running on an FPGA.
- Portfolio: https://genwealth.world

## License

MIT. See [LICENSE](LICENSE).
