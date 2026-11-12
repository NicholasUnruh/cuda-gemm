#pragma once
// Error checks, timing and the cuBLASLt FP8 reference, shared by bench.cu
// and numerics.cu.
#include <cublasLt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                   \
  do {                                                  \
    cudaError_t e = (x);                                \
    if (e != cudaSuccess) {                             \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                   \
      exit(2);                                          \
    }                                                   \
  } while (0)
#define CUBLAS_CHECK(x)                                            \
  do {                                                             \
    if ((x) != CUBLAS_STATUS_SUCCESS) {                            \
      fprintf(stderr, "%s:%d cuBLAS error\n", __FILE__, __LINE__); \
      exit(2);                                                     \
    }                                                              \
  } while (0)

inline const int kWarmup = 2;
inline const int kReps = 10;

// Mean milliseconds per call of run() over kReps, after kWarmup calls.
template <class Run>
float time_ms(Run run) {
  for (int i = 0; i < kWarmup; ++i) run();
  CUDA_CHECK(cudaGetLastError());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kReps; ++i) run();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / kReps;
}

// FP8 reference: cuBLASLt on the same E4M3 codes, FP32 accumulation and C.
// cuBLASLt FP8 needs both operands K-major ("TN"). Column-major, its A is our
// Bt (K x N, transposed) and its B is our A (K x M), giving C^T = row-major C.
// The descriptors and algorithm are chosen once per size, outside the timing.
struct LtFp8 {
  cublasLtHandle_t lt;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulHeuristicResult_t algo;
  void *workspace;
  size_t workspace_bytes = 32 << 20;

  LtFp8(int M, int N, int K) {
    CUBLAS_CHECK(cublasLtCreate(&lt));
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t t = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, N, M, N));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
        sizeof(workspace_bytes)));
    int found = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, a, b, c, c, pref, 1,
                                                &algo, &found));
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    if (found == 0) {
      fprintf(stderr, "cuBLASLt has no FP8 algorithm for this GPU\n");
      exit(2);
    }
  }
  ~LtFp8() {
    cudaFree(workspace);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulDescDestroy(op);
    cublasLtDestroy(lt);
  }
  void run(float alpha, const uint8_t *A, const uint8_t *Bt, float beta,
           float *C) {
    CUBLAS_CHECK(cublasLtMatmul(lt, op, &alpha, Bt, a, A, b, &beta, C, c, C, c,
                                &algo.algo, workspace, workspace_bytes, 0));
  }
};

