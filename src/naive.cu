#include "gemm.cuh"

// One thread per output element. Consecutive threads take consecutive rows, so
// a warp's loads of A and stores of C stride by K and N (uncoalesced) while its
// loads of B hit one address. This is the baseline row.
__global__ void naive_kernel(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

void gemm_naive(int M, int N, int K, float alpha, const float *A,
                const float *B, float beta, float *C) {
  dim3 block(32, 32);
  dim3 grid((M + 31) / 32, (N + 31) / 32);
  naive_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
