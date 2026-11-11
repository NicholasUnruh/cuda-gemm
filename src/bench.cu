// Times one kernel against cuBLAS on square sizes and checks its result.
// Usage: bench <kernel> [size ...]   Prints CSV; exits 1 on a wrong result.
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "gemm.cuh"

#define CUDA_CHECK(x)                                                         \
  do {                                                                        \
    cudaError_t e = (x);                                                      \
    if (e != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(2);                                                                \
    }                                                                         \
  } while (0)
#define CUBLAS_CHECK(x)                                                       \
  do {                                                                        \
    if ((x) != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "%s:%d cuBLAS error\n", __FILE__, __LINE__);            \
      exit(2);                                                                \
    }                                                                         \
  } while (0)

struct Kernel {
  const char *name;
  GemmFn fn;
};

// Add each new kernel here.
static const Kernel kKernels[] = {
    {"naive", gemm_naive},
};

static const int kWarmup = 2;
static const int kReps = 10;
static const float kTolerance = 1e-4f;  // max |C - ref| / max |ref|, FP32 kernels

static cublasHandle_t g_handle;

// cuBLAS is column-major: row-major C = A B is column-major C^T = B^T A^T.
static void gemm_cublas(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C) {
  CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           B, N, A, K, &beta, C, N));
}

// Mean milliseconds per call over kReps, after kWarmup calls.
static float time_ms(GemmFn fn, int n, const float *A, const float *B, float *C) {
  for (int i = 0; i < kWarmup; ++i) fn(n, n, n, 1.f, A, B, 0.f, C);
  CUDA_CHECK(cudaGetLastError());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < kReps; ++i) fn(n, n, n, 1.f, A, B, 0.f, C);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / kReps;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <kernel> [size ...]\n", argv[0]);
    return 2;
  }
  GemmFn fn = nullptr;
  for (const Kernel &k : kKernels)
    if (strcmp(k.name, argv[1]) == 0) fn = k.fn;
  if (!fn) {
    fprintf(stderr, "unknown kernel %s\n", argv[1]);
    return 2;
  }
  std::vector<int> sizes;
  for (int i = 2; i < argc; ++i) sizes.push_back(atoi(argv[i]));
  if (sizes.empty()) sizes = {256, 512, 1024, 2048, 4096};

  CUBLAS_CHECK(cublasCreate(&g_handle));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("# %s\nkernel,n,ms,gflops,cublas_gflops,pct_cublas,max_rel_err\n", prop.name);

  bool ok = true;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (int n : sizes) {
    size_t count = size_t(n) * n, bytes = count * sizeof(float);
    std::vector<float> hA(count), hB(count), hC(count), hRef(count);
    for (float &x : hA) x = dist(rng);
    for (float &x : hB) x = dist(rng);
    float *A, *B, *C, *Ref;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));
    CUDA_CHECK(cudaMalloc(&Ref, bytes));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

    float ms = time_ms(fn, n, A, B, C);
    float ref_ms = time_ms(gemm_cublas, n, A, B, Ref);
    CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRef.data(), Ref, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    for (size_t i = 0; i < count; ++i) {
      max_err = fmax(max_err, fabs(double(hC[i]) - hRef[i]));
      max_ref = fmax(max_ref, fabs(double(hRef[i])));
    }
    double rel = max_err / max_ref;
    double flops = 2.0 * n * n * n;
    double gflops = flops / (ms * 1e6), ref_gflops = flops / (ref_ms * 1e6);
    printf("%s,%d,%.4f,%.1f,%.1f,%.1f,%.2e\n", argv[1], n, ms, gflops,
           ref_gflops, 100.0 * gflops / ref_gflops, rel);
    if (!(rel <= kTolerance)) ok = false;

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(Ref));
  }
  CUBLAS_CHECK(cublasDestroy(g_handle));
  if (!ok) fprintf(stderr, "FAIL: error above %.0e\n", kTolerance);
  return ok ? 0 : 1;
}
