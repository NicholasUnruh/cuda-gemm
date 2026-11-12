#include "gemm.cuh"

// Same arithmetic as naive; only the thread-to-output mapping changes.
// threadIdx.x now walks columns, so the 32 threads of a warp read 32
// consecutive floats of B (one 128-byte transaction) and write 32 consecutive
// floats of C. All threads in a warp share a row, so their A load is a single
// broadcast address.
__global__ void coalesced_kernel(int M, int N, int K, float alpha,
                                 const float *A, const float *B, float beta,
                                 float *C) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

void gemm_coalesced(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  coalesced_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
