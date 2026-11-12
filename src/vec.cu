#include "gemm.cuh"

constexpr int BM = 128, BN = 128, BK = 8;
constexpr int TM = 8, TN = 8;
constexpr int kThreads = (BM / TM) * (BN / TN);  // 256

// The reg2d kernel with 128-bit memory operations. Each thread copies one
// float4 of A and one of B per K step (instead of 4 scalar loads each), and A
// is stored transposed (As[k][m]) so a thread's 8 values of A for one k are
// contiguous and come back in two 128-bit shared loads, like its 8 values of
// B. C is written with float4 stores. Fewer, wider instructions for the same
// bytes free issue slots for FMAs.
__global__ __launch_bounds__(kThreads) void vec_kernel(
    int M, int N, int K, float alpha, const float *A, const float *B,
    float beta, float *C) {
  __shared__ float4 As4[BK * BM / 4];  // As[k][m], transposed, BK x BM
  __shared__ float4 Bs4[BK * BN / 4];  // Bs[k][n], BK x BN
  float *As = reinterpret_cast<float *>(As4);
  const int tid = threadIdx.x;
  const int threadRow = tid / (BN / TN);
  const int threadCol = tid % (BN / TN);
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // A tile 128x8: each row is 2 float4s, one per thread. B tile 8x128: each
  // row is 32 float4s, so 256 threads cover all 8 rows in one pass.
  const int aRow = tid / (BK / 4), aCol = tid % (BK / 4);
  const int bRow = tid / (BN / 4), bCol = tid % (BN / 4);

  float acc[TM][TN] = {};
  float regM[TM], regN[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    float4 a = reinterpret_cast<const float4 *>(&A[aRow * K])[aCol];
    As[(aCol * 4 + 0) * BM + aRow] = a.x;
    As[(aCol * 4 + 1) * BM + aRow] = a.y;
    As[(aCol * 4 + 2) * BM + aRow] = a.z;
    As[(aCol * 4 + 3) * BM + aRow] = a.w;
    Bs4[bRow * (BN / 4) + bCol] =
        reinterpret_cast<const float4 *>(&B[bRow * N])[bCol];
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; i += 4) {
        float4 m = As4[(k * BM + threadRow * TM + i) / 4];
        regM[i] = m.x, regM[i + 1] = m.y, regM[i + 2] = m.z, regM[i + 3] = m.w;
      }
      for (int j = 0; j < TN; j += 4) {
        float4 n = Bs4[(k * BN + threadCol * TN + j) / 4];
        regN[j] = n.x, regN[j + 1] = n.y, regN[j + 2] = n.z, regN[j + 3] = n.w;
      }
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; j += 4) {
      float4 *p = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + i) * N + threadCol * TN + j]);
      float4 c = *p;
      c.x = alpha * acc[i][j] + beta * c.x;
      c.y = alpha * acc[i][j + 1] + beta * c.y;
      c.z = alpha * acc[i][j + 2] + beta * c.z;
      c.w = alpha * acc[i][j + 3] + beta * c.w;
      *p = c;
    }
}

void gemm_vec(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
  dim3 grid(N / BN, M / BM);
  vec_kernel<<<grid, kThreads>>>(M, N, K, alpha, A, B, beta, C);
}
