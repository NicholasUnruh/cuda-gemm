# Running on a GPU

`make check` compiles for sm_75, sm_80 and sm_89 without a GPU; timing needs one.

## RTX 4060

The README numbers come from an RTX 4060 (sm_89, CUDA 13.3) under WSL2:

```sh
make bench ARCH=sm_89 KERNEL=vec
```

Time only when `nvidia-smi` shows the GPU idle; anything else on it slows cuBLAS and the kernel unevenly. Raw runs are in [results/rtx4060.csv](results/rtx4060.csv).

## Colab (free T4)

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

`make profile` collects the roofline counters for every kernel at 4096 as CSV; `uv run scripts/roofline.py <csv> docs/roofline.png "<GPU>"` plots them ([RTX 4060 counters](results/ncu-rtx4060.csv)). On Windows/WSL the driver refuses counter access (`ERR_NVGPUCTRPERM`) until NVIDIA Control Panel → Developer → Manage GPU Performance Counters allows all users.

Open the report in Nsight Compute. The memory-throughput and occupancy sections explain most of the gap to cuBLAS at each step.

## Recording a result

Copy the 4096 row into the README table with the GPU name. Keep T4 numbers apart from numbers on any other GPU.
