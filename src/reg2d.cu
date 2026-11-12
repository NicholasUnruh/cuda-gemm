#include "gemm.cuh"

constexpr int BM = 128, BN = 128, BK = 8;  // block tile of C, K step
constexpr int TM = 8, TN = 8;              // per-thread tile of C
constexpr int kThreads = (BM / TM) * (BN / TN);  // 256

// Each thread computes an 8x8 block of C held in 64 registers. Per k it loads
// 8 floats of A and 8 of B from shared memory into registers and does 64 FMAs:
// 16 shared loads per 64 FMAs, against 2 per 1 in the smem step. Shared-memory
// bandwidth stops being the limit and the FMA pipe does more of the work.
__global__ __launch_bounds__(kThreads) void reg2d_kernel(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
  __shared__ float As[BM * BK];  // row-major BM x BK
  __shared__ float Bs[BK * BN];  // row-major BK x BN
  const int tid = threadIdx.x;
  const int threadRow = tid / (BN / TN);  // 0..15
  const int threadCol = tid % (BN / TN);  // 0..15
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // Load mapping: 256 threads copy the 128x8 A tile in 4 passes of 32 rows
  // and the 8x128 B tile in 4 passes of 2 rows.
  const int aRow = tid / BK, aCol = tid % BK;
  const int bRow = tid / BN, bCol = tid % BN;
  constexpr int aStride = kThreads / BK, bStride = kThreads / BN;

  float acc[TM][TN] = {};
  float regM[TM], regN[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int r = 0; r < BM; r += aStride)
      As[(aRow + r) * BK + aCol] = A[(aRow + r) * K + aCol];
    for (int r = 0; r < BK; r += bStride)
      Bs[(bRow + r) * BN + bCol] = B[(bRow + r) * N + bCol];
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + k];
      for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + threadCol * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      float &c = C[(threadRow * TM + i) * N + threadCol * TN + j];
      c = alpha * acc[i][j] + beta * c;
    }
}

void gemm_reg2d(int M, int N, int K, float alpha, const float *A,
                const float *B, float beta, float *C) {
  dim3 grid(N / BN, M / BM);
  reg2d_kernel<<<grid, kThreads>>>(M, N, K, alpha, A, B, beta, C);
}
