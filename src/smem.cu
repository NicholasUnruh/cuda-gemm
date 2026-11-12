#include "gemm.cuh"

constexpr int kTile = 32;

// Each 32x32 block computes a 32x32 tile of C. It walks K in steps of 32: the
// block cooperatively copies a 32x32 tile of A and of B into shared memory
// (one coalesced float per thread each), syncs, then every thread does 32 FMAs
// from shared memory. Each global float is now read by the block once instead
// of by 32 threads, cutting global traffic per FMA by 32x.
__global__ void smem_kernel(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * kTile + ty;
  int col = blockIdx.x * kTile + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += kTile) {
    As[ty][tx] = A[row * K + k0 + tx];
    Bs[ty][tx] = B[(k0 + ty) * N + col];
    __syncthreads();
    // As[ty][k] is one address per warp (broadcast); Bs[k][tx] is 32
    // consecutive words (one per bank). No bank conflicts.
    for (int k = 0; k < kTile; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

void gemm_smem(int M, int N, int K, float alpha, const float *A,
               const float *B, float beta, float *C) {
  dim3 block(kTile, kTile);
  dim3 grid(N / kTile, M / kTile);
  smem_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
