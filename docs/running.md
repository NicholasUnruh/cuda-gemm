# Running on a GPU

This laptop has no NVIDIA GPU. `make check` compiles locally; timing needs a GPU.

## RTX 4060 over SSH (bro-run)

The README numbers come from an RTX 4060 (sm_89, CUDA 13.3) in WSL on a shared PC. `bro-run` mirrors the repo there and runs one command:

```sh
bro-run -- make bench ARCH=sm_89 KERNEL=vec
```

The PC is someone's gaming machine: time only when `nvidia-smi` shows it idle, since a game on the GPU slows cuBLAS and the kernel unevenly. Raw runs are in [results/rtx4060.csv](results/rtx4060.csv).

## Colab (free T4)

1. Runtime → Change runtime type → T4 GPU.
2. While the repo is private, upload a zip (`git archive -o cuda-gemm.zip HEAD`) and unzip it, or clone with a fine-grained read-only token.
3. In a cell:

```sh
!nvidia-smi
!cd cuda-gemm && make bench KERNEL=naive CUDA_HOME=/usr/local/cuda ARCH=sm_75
```

## Profiling

```sh
ncu --set full -o build/naive ./build/bench naive 4096
```

Open the report in Nsight Compute. The memory-throughput and occupancy sections explain most of the gap to cuBLAS at each step.

## Recording a result

Copy the 4096 row into the README table with the GPU name. Keep T4 numbers apart from numbers on any other GPU.
