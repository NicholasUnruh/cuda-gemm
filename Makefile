.PHONY: check bench profile toolchain clean

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

# Needs a GPU and Nsight Compute: counters for one launch of every kernel at
# 4096, as raw CSV on stdout (feed to scripts/roofline.py). ncu locks clocks to
# base, so its times are slower than `make bench`; peaks come from the same run.
NCU ?= $(CUDA_HOME)/bin/ncu
KERNELS := naive coalesced smem reg2d vec wmma
STALLS := long_scoreboard mio_throttle short_scoreboard barrier math_pipe_throttle lg_throttle not_selected
comma := ,
empty :=
space := $(empty) $(empty)
METRIC_LIST := gpu__time_duration.sum dram__bytes_read.sum dram__bytes_write.sum lts__t_bytes.sum \
  dram__bytes.sum.peak_sustained dram__cycles_elapsed.avg.per_second sm__cycles_elapsed.avg.per_second \
  sm__sass_thread_inst_executed_op_ffma_pred_on.sum sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained \
  sm__ops_path_tensor_src_fp16_dst_fp32.sum sm__ops_path_tensor_src_fp16_dst_fp32.sum.peak_sustained \
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  l1tex__throughput.avg.pct_of_peak_sustained_active sm__warps_active.avg.pct_of_peak_sustained_active \
  launch__registers_per_thread smsp__inst_executed.sum \
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active \
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
  $(foreach s,$(STALLS),smsp__average_warps_issue_stalled_$(s)_per_issue_active.ratio)
METRICS := $(subst $(space),$(comma),$(strip $(METRIC_LIST)))
profile: build/bench
	for k in $(KERNELS); do \
	  $(NCU) --csv --page raw --metrics $(METRICS) -k regex:^$${k}_kernel$$ \
	    --launch-skip 2 --launch-count 1 --log-file build/ncu-$$k.log ./build/bench $$k 4096 >/dev/null; \
	  grep '^"' build/ncu-$$k.log || { cat build/ncu-$$k.log >&2; exit 1; }; \
	done

toolchain:
	test -x $(LOCAL_CUDA)/bin/nvcc || MAMBA_ROOT_PREFIX=$(HOME)/.local/opt/mamba \
		timeout 1200 micromamba create -y -p $(LOCAL_CUDA) -c conda-forge \
		cuda-nvcc=12.9 cuda-cudart-dev=12.9 libcublas-dev=12.9 gxx_linux-64=13

clean:
	rm -rf build
