#include <mma.h>

#include "gemm.cuh"

using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 32;     // block tile of C, K step
constexpr int WM = 64, WN = 32;                // warp tile of C
constexpr int kWarps = (BM / WM) * (BN / WN);  // 8 warps, 2 x 4
constexpr int kThreads = kWarps * 32;          // 256
constexpr int FM = WM / 16, FN = WN / 16;      // 4 x 2 16x16 fragments per warp
constexpr int kPad = 8;                        // as in wmma.cu
constexpr int LDA = BK + kPad, LDB = BN + kPad;
constexpr int kStages = 2;
// Blocks are launched in groups of kGroup block rows, column by column inside a
// group, so the blocks resident at one time share A rows and B columns in L2.
constexpr int kGroup = 8;

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

// Step 6 with the K loop software-pipelined: two shared-memory buffers per
// operand. While the warps multiply tile k out of one buffer, cp.async copies
// tile k+1 into the other, so tensor cores no longer sit idle during loads.
__global__ void __launch_bounds__(kThreads)
    cpasync_kernel(int M, int N, int K, float alpha, const __half *A,
                   const __half *B, float beta, float *C) {
  __shared__ __align__(128) __half As[kStages][BM * LDA];
  __shared__ __align__(128) __half Bs[kStages][BK * LDB];
  const int tid = threadIdx.x, warp = tid / 32;
  const int warpRow = warp / (BN / WN), warpCol = warp % (BN / WN);
  // Block index -> (block row, block column), grouped as described at kGroup.
  const int cols = N / BN, rows = M / BM, perGroup = kGroup * cols;
  const int first = blockIdx.x / perGroup * kGroup;
  const int height = min(kGroup, rows - first), inGroup = blockIdx.x % perGroup;
  const int blockRow = first + inGroup % height, blockCol = inGroup / height;
  A += blockRow * BM * K;
  B += blockCol * BN;
  C += blockRow * BM * N + blockCol * BN;

  // Issues the copies of K tile t into buffer s: 512 16-byte chunks per
  // operand, two per thread.
  auto load_tile = [&](int t, int s) {
    const __half *a = A + t * BK, *b = B + t * BK * N;
    for (int c = tid; c < BM * BK / 8; c += kThreads) {
      int r = c / (BK / 8), col = c % (BK / 8) * 8;
      cp_async16(&As[s][r * LDA + col], &a[r * K + col]);
    }
    for (int c = tid; c < BK * BN / 8; c += kThreads) {
      int r = c / (BN / 8), col = c % (BN / 8) * 8;
      cp_async16(&Bs[s][r * LDB + col], &b[r * N + col]);
    }
    cp_async_commit();
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  const int tiles = K / BK;
  load_tile(0, 0);
  for (int t = 0; t < tiles; ++t) {
    const int s = t % kStages;
    cp_async_wait<0>();  // this thread's copies of tile t have landed
    // Everyone's copies of tile t are visible, and every warp has finished
    // tile t-1, so its buffer can be refilled with tile t+1 while tile t is
    // multiplied.
    __syncthreads();
    if (t + 1 < tiles) load_tile(t + 1, (t + 1) % kStages);
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[FN];
      for (int i = 0; i < FM; ++i)
        wmma::load_matrix_sync(a[i], &As[s][(warpRow * WM + i * 16) * LDA + kk],
                               LDA);
      for (int j = 0; j < FN; ++j)
        wmma::load_matrix_sync(b[j], &Bs[s][kk * LDB + warpCol * WN + j * 16],
                               LDB);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j)
          wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }

  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) {
      float *c = &C[(warpRow * WM + i * 16) * N + warpCol * WN + j * 16];
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> old;
      wmma::load_matrix_sync(old, c, N, wmma::mem_row_major);
      for (int e = 0; e < old.num_elements; ++e)
        old.x[e] = alpha * acc[i][j].x[e] + beta * old.x[e];
      wmma::store_matrix_sync(c, old, N, wmma::mem_row_major);
    }
}

void gemm_cpasync(int M, int N, int K, float alpha, const __half *A,
                  const __half *B, float beta, float *C) {
  cpasync_kernel<<<(M / BM) * (N / BN), kThreads>>>(M, N, K, alpha, A, B, beta, C);
}
