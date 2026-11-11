.PHONY: check bench toolchain clean

# Local: the micromamba CUDA env from `make toolchain` (no GPU; compiles only).
# Colab or a GPU box: CUDA_HOME=/usr/local/cuda, system g++.
LOCAL_CUDA := $(HOME)/.local/opt/cuda-12.9
CUDA_HOME ?= $(if $(wildcard $(LOCAL_CUDA)/bin/nvcc),$(LOCAL_CUDA),/usr/local/cuda)
NVCC := $(CUDA_HOME)/bin/nvcc
CCBIN := $(wildcard $(CUDA_HOME)/bin/x86_64-conda-linux-gnu-g++)
# T4 (Colab free) is sm_75, L4 is sm_89, A100 is sm_80.
ARCH ?= sm_75
FLAGS := -O3 -std=c++17 -lineinfo $(if $(CCBIN),-ccbin $(CCBIN))
SOURCES := $(wildcard src/*.cu)

build/bench: $(SOURCES) src/gemm.cuh
	mkdir -p build
	$(NVCC) $(FLAGS) -arch=$(ARCH) -o $@ $(SOURCES) -lcublas

# Compiles every kernel for T4, A100 and L4. Needs no GPU. Run before every commit.
check:
	mkdir -p build
	$(NVCC) $(FLAGS) -gencode arch=compute_75,code=sm_75 \
		-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89 \
		-o build/bench-check $(SOURCES) -lcublas

# Needs an NVIDIA GPU: make bench KERNEL=naive
KERNEL ?= naive
bench: build/bench
	./build/bench $(KERNEL)

toolchain:
	test -x $(LOCAL_CUDA)/bin/nvcc || MAMBA_ROOT_PREFIX=$(HOME)/.local/opt/mamba \
		timeout 1200 micromamba create -y -p $(LOCAL_CUDA) -c conda-forge \
		cuda-nvcc=12.9 cuda-cudart-dev=12.9 libcublas-dev=12.9 gxx_linux-64=13

clean:
	rm -rf build
