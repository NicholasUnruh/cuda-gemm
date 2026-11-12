#include <mma.h>

#include "gemm.cuh"

using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 32;     // block tile of C, K step
constexpr int WM = 64, WN = 32;                // warp tile of C
constexpr int kWarps = (BM / WM) * (BN / WN);  // 8 warps, 2 x 4
constexpr int kThreads = kWarps * 32;          // 256
constexpr int FM = WM / 16, FN = WN / 16;      // 4 x 2 16x16 fragments per warp
// Row padding in halves: shifts each shared row by 16 bytes so the 8 rows a
// fragment load touches fall in different banks. Keeps rows 32-byte aligned
// at fragment offsets, as load_matrix_sync requires.
constexpr int kPad = 8;
constexpr int LDA = BK + kPad, LDB = BN + kPad;

// Tensor cores: each warp issues 16x16x16 matrix multiply-accumulates (FP16
// inputs, FP32 accumulators) instead of scalar FMAs. A block stages a 128x32
// tile of A and a 32x128 tile of B in shared memory with 128-bit loads (8
// halves per thread per load); each warp then multiplies its 64x32 slice of C
// as 4x2 fragments, reusing every A fragment twice and every B fragment four
// times from registers.
__global__ void __launch_bounds__(kThreads)
    wmma_kernel(int M, int N, int K, float alpha, const __half *A,
                const __half *B, float beta, float *C) {
  __shared__ __align__(128) __half As[BM * LDA];
  __shared__ __align__(128) __half Bs[BK * LDB];
  const int tid = threadIdx.x, warp = tid / 32;
  const int warpRow = warp / (BN / WN), warpCol = warp % (BN / WN);
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
  for (int i = 0; i < FM; ++i)
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    // 128x32 A tile = 512 uint4 chunks of 8 halves; 32x128 B tile likewise.
    for (int c = tid; c < BM * BK / 8; c += kThreads) {
      int r = c / (BK / 8), col = c % (BK / 8) * 8;
      *reinterpret_cast<uint4 *>(&As[r * LDA + col]) =
          *reinterpret_cast<const uint4 *>(&A[r * K + col]);
    }
    for (int c = tid; c < BK * BN / 8; c += kThreads) {
      int r = c / (BN / 8), col = c % (BN / 8) * 8;
      *reinterpret_cast<uint4 *>(&Bs[r * LDB + col]) =
          *reinterpret_cast<const uint4 *>(&B[r * N + col]);
    }
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[FN];
      for (int i = 0; i < FM; ++i)
        wmma::load_matrix_sync(a[i], &As[(warpRow * WM + i * 16) * LDA + kk],
                               LDA);
      for (int j = 0; j < FN; ++j)
        wmma::load_matrix_sync(b[j], &Bs[kk * LDB + warpCol * WN + j * 16],
                               LDB);
      for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j)
          wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }

  // Accumulator fragments of one type share an element layout, so C can be
  // scaled element by element in fragment form.
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

void gemm_wmma(int M, int N, int K, float alpha, const __half *A,
               const __half *B, float beta, float *C) {
  dim3 grid(N / BN, M / BM);
  wmma_kernel<<<grid, kThreads>>>(M, N, K, alpha, A, B, beta, C);
}
