# Running on a GPU

`make check` compiles every kernel for sm_75, sm_80 and sm_89 without a GPU. Timing and profiling need one.

## Benchmark

The README numbers come from an RTX 4060 (sm_89, CUDA 13.3, WSL2):

```sh
make bench ARCH=sm_89 KERNEL=vec
```

`bench` times each size against cuBLAS on the same inputs and prints CSV: milliseconds, GFLOPS, % of cuBLAS and `max_rel_err`. It exits non-zero when the error exceeds the tolerance (1e-4 for FP32 kernels, 1e-3 for the FP16-input `wmma` and `cpasync`).

Time only when `nvidia-smi` shows the GPU idle; other work on the GPU slows cuBLAS and the kernel unevenly. Raw runs are in [results/rtx4060.csv](results/rtx4060.csv).

## Google Colab (T4)

1. Runtime → Change runtime type → T4 GPU.
2. In a cell: `!git clone https://github.com/NicholasUnruh/cuda-gemm`.
3. Then:

```sh
!nvidia-smi
!cd cuda-gemm && make bench KERNEL=naive CUDA_HOME=/usr/local/cuda ARCH=sm_75
```

## Profiling

```sh
ncu --set full -o build/naive ./build/bench naive 4096
```

The memory-throughput and occupancy sections of the report explain most of the gap to cuBLAS at each step.

`make profile` collects the roofline counters for every kernel at 4096³ as CSV, and the scripts plot them (`counters.py` titles its figure RTX 4060):

```sh
mkdir -p build && make profile ARCH=sm_89 > build/ncu.csv
uv run scripts/roofline.py build/ncu.csv docs/roofline.png "RTX 4060"
uv run scripts/counters.py build/ncu.csv docs/img/counters.png
uv run scripts/tiling.py docs/img/tiling.png   # reads docs/results/*.csv
```

Each plotting script takes an optional `dark` theme and writes `.webp` when the output ends in it; `counters.py` and `tiling.py` also take a JSON output path. `tiling.py` fails if a cited source line changed, and prints the global load requests and sectors its tiling implies next to the measured ones.

The RTX 4060 counters are in [results/ncu-rtx4060.csv](results/ncu-rtx4060.csv). On Windows and WSL2 the driver refuses counter access (`ERR_NVGPUCTRPERM`) until NVIDIA Control Panel → Developer → Manage GPU Performance Counters allows all users.

## Recording a result

Copy the median 4096 row into the README table with the GPU name. Keep numbers from different GPUs in separate tables.
