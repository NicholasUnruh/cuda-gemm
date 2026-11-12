#pragma once

// Pieces shared by the pipelined kernels (steps 7-9).

// 16-byte copy from global to shared memory that does not wait for the data.
// sm_80+ issues cp.async (global -> L2 -> shared, no registers); older GPUs
// fall back to an ordinary load and store, which is correct but not async.
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
#if __CUDA_ARCH__ >= 800
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
               "l"(gmem));
#else
  *reinterpret_cast<uint4 *>(smem) = *reinterpret_cast<const uint4 *>(gmem);
#endif
}
// Closes the group of copies issued since the last commit.
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
// Waits until at most n committed groups are still in flight.
template <int n>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

// Block index -> (block row, block column) of C. Blocks go column by column
// inside groups of `group` block rows, so the blocks resident at one time
// share A rows and B columns in L2 (step 7).
__device__ __forceinline__ void grouped_tile(int rows, int cols, int group,
                                             int &blockRow, int &blockCol) {
  const int perGroup = group * cols;
  const int first = blockIdx.x / perGroup * group;
  const int height = min(group, rows - first), inGroup = blockIdx.x % perGroup;
  blockRow = first + inGroup % height;
  blockCol = inGroup / height;
}
