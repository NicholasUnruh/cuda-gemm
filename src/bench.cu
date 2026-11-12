// Times one kernel against cuBLAS on square sizes and checks its result.
// Usage: bench <kernel> [size ...]   Prints CSV; exits 1 on a wrong result.
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_fp8.h>

#include "gemm.cuh"
#include "quant.cuh"
#include "util.cuh"

// How bench quantizes the FP32 inputs for an fp8_fn kernel.
enum class Fp8Data {
  kTensor,  // E4M3, one scale per tensor (alpha)
  kMxfp8,   // E4M3 elements, E8M0 scale per 32
  kMxfp4,   // E2M1 elements re-encoded as E4M3, E8M0 scale per 32
};

// Exactly one of fn (FP32 inputs), half_fn (FP16) and fp8_fn (E4M3) is set.
struct Kernel {
  const char *name;
  GemmFn fn;
  GemmHalfFn half_fn;
  GemmFp8Fn fp8_fn;
  Fp8Data data;
};

// Add each new kernel here.
// clang-format off
static const Kernel kKernels[] = {
    {"naive", gemm_naive, nullptr},
    {"coalesced", gemm_coalesced, nullptr},
    {"smem", gemm_smem, nullptr},
    {"reg2d", gemm_reg2d, nullptr},
    {"vec", gemm_vec, nullptr},
    {"wmma", nullptr, gemm_wmma},
    {"cpasync", nullptr, gemm_cpasync},
    {"fp8", nullptr, nullptr, gemm_fp8, Fp8Data::kTensor},
    {"mxfp8", nullptr, nullptr, gemm_mxfp8, Fp8Data::kMxfp8},
    {"mxfp4", nullptr, nullptr, gemm_mxfp8, Fp8Data::kMxfp4},
};
// clang-format on

// max |C - ref| / max |ref|. FP32 kernels are checked against cublasSgemm.
// FP16-input kernels are checked against cublasGemmEx on the same FP16
// inputs with FP32 accumulation: summation order and the tensor cores'
// internal rounding differ, so they get a 10x looser bound. FP8 kernels are
// checked against exact (FP64) dot products of their E4M3 inputs instead,
// because the FP8 tensor cores truncate partial sums (step 8 note) and
// cuBLASLt is itself off by about 1e-4; its error is printed as a comment.
static const float kTolerance = 1e-4f;
static const float kHalfTolerance = 1e-3f;

static cublasHandle_t g_handle;

// cuBLAS is column-major: row-major C = A B is column-major C^T = B^T A^T.
static void gemm_cublas(int M, int N, int K, float alpha, const float *A,
                        const float *B, float beta, float *C) {
  CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           B, N, A, K, &beta, C, N));
}

// FP16 A and B, FP32 accumulation and C: the reference for tensor-core kernels.
static void gemm_cublas_half(int M, int N, int K, float alpha, const __half *A,
                             const __half *B, float beta, float *C) {
  CUBLAS_CHECK(cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            B, CUDA_R_16F, N, A, CUDA_R_16F, K, &beta, C,
                            CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

// Error of C against exact dot products of the quantized inputs, on 4,096
// sampled entries: max |C - exact| / max |exact|. A and Bt are E4M3 codes;
// sa and sb are their MX block scales, or null for per-tensor scaling.
static double exact_err(int n, float alpha, const uint8_t *A, const uint8_t *sa,
                        const uint8_t *Bt, const uint8_t *sb, const float *C) {
  size_t count = size_t(n) * n;
  std::vector<uint8_t> a(count), b(count), as(count / 32, 127),
      bs(count / 32, 127);  // 127 is 2^0
  std::vector<float> c(count);
  CUDA_CHECK(cudaMemcpy(a.data(), A, count, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b.data(), Bt, count, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(c.data(), C, count * 4, cudaMemcpyDeviceToHost));
  if (sa) {
    CUDA_CHECK(cudaMemcpy(as.data(), sa, count / 32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bs.data(), sb, count / 32, cudaMemcpyDeviceToHost));
  }
  double lut[256];
  for (int v = 0; v < 256; ++v)
    lut[v] = __half2float(__nv_cvt_fp8_to_halfraw(uint8_t(v), __NV_E4M3));
  std::mt19937 rng(7);
  double max_err = 0, max_ref = 0;
  for (int s = 0; s < 4096; ++s) {
    int i = rng() % n, j = rng() % n;
    double exact = 0;
    for (int k = 0; k < n; ++k) {
      size_t ia = size_t(i) * n + k, ib = size_t(j) * n + k;
      exact += std::ldexp(lut[a[ia]] * lut[b[ib]],
                          as[ia / 32] + bs[ib / 32] - 254);
    }
    exact *= alpha;
    max_err = fmax(max_err, fabs(c[size_t(i) * n + j] - exact));
    max_ref = fmax(max_ref, fabs(exact));
  }
  return max_err / max_ref;
}

int run_numerics();  // numerics.cu

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <kernel> [size ...] | numerics\n", argv[0]);
    return 2;
  }
  if (strcmp(argv[1], "numerics") == 0) return run_numerics();
  const Kernel *kernel = nullptr;
  for (const Kernel &k : kKernels)
    if (strcmp(k.name, argv[1]) == 0) kernel = &k;
  if (!kernel) {
    fprintf(stderr, "unknown kernel %s\n", argv[1]);
    return 2;
  }
  std::vector<int> sizes;
  for (int i = 2; i < argc; ++i) {
    sizes.push_back(atoi(argv[i]));
    if (sizes.back() <= 0 || sizes.back() % 128 != 0) {
      fprintf(stderr, "size %s: must be a positive multiple of 128\n", argv[i]);
      return 2;
    }
  }
  if (sizes.empty()) sizes = {256, 512, 1024, 2048, 4096};
  const float tolerance = kernel->fn ? kTolerance : kHalfTolerance;

  CUBLAS_CHECK(cublasCreate(&g_handle));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (kernel->fp8_fn && prop.major * 10 + prop.minor < 89) {
    fprintf(stderr, "%s needs FP8 tensor cores (sm_89 or newer)\n", argv[1]);
    return 2;
  }
  printf("# %s\nkernel,n,ms,gflops,cublas_gflops,pct_cublas,max_rel_err\n",
         prop.name);

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
    // beta is 0, but the kernels still read C: start from zeros, not garbage.
    CUDA_CHECK(cudaMemset(C, 0, bytes));
    CUDA_CHECK(cudaMemset(Ref, 0, bytes));
    CUDA_CHECK(cudaMemcpy(A, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB.data(), bytes, cudaMemcpyHostToDevice));

    float ms, ref_ms;
    double exact = -1;  // FP8 only: error against exact dot products
    if (kernel->fn) {
      ms = time_ms([&] { kernel->fn(n, n, n, 1.f, A, B, 0.f, C); });
      ref_ms = time_ms([&] { gemm_cublas(n, n, n, 1.f, A, B, 0.f, Ref); });
    } else if (kernel->fp8_fn) {
      // E4M3 codes (and MX scales), quantized once outside the timed loop.
      // B's random values are used as Bt (N x K): FP8 needs B K-major.
      // The reference is per-tensor cuBLASLt FP8 on the same codes: cuBLASLt
      // has no MX block scaling on Ada, so for MX it is a timing reference.
      uint8_t *Aq, *Bq, *sa = nullptr, *sb = nullptr, *packed;
      unsigned *amax;
      CUDA_CHECK(cudaMalloc(&Aq, count));
      CUDA_CHECK(cudaMalloc(&Bq, count));
      CUDA_CHECK(cudaMalloc(&packed, count / 2));
      CUDA_CHECK(cudaMalloc(&amax, 2 * sizeof(unsigned)));
      float alpha = 1.f;
      if (kernel->data == Fp8Data::kTensor) {
        quant_amax(A, count, amax);
        quant_amax(B, count, amax + 1);
        quant_e4m3_tensor(A, count, amax, Aq);
        quant_e4m3_tensor(B, count, amax + 1, Bq);
        unsigned bits[2];
        CUDA_CHECK(
            cudaMemcpy(bits, amax, sizeof(bits), cudaMemcpyDeviceToHost));
        float am[2];
        memcpy(am, bits, sizeof(am));
        // Each code is x * 448 / amax, so the product carries (448 / amax)^2.
        alpha = float(double(am[0]) / 448.0 * am[1] / 448.0);
      } else {
        CUDA_CHECK(cudaMalloc(&sa, count / 32));
        CUDA_CHECK(cudaMalloc(&sb, count / 32));
        if (kernel->data == Fp8Data::kMxfp8) {
          quant_mx(A, count, false, Aq, sa);
          quant_mx(B, count, false, Bq, sb);
        } else {
          quant_mx(A, count, true, packed, sa);
          e2m1_to_e4m3(packed, count, Aq);
          quant_mx(B, count, true, packed, sb);
          e2m1_to_e4m3(packed, count, Bq);
        }
      }
      LtFp8 lt(n, n, n);
      ms = time_ms(
          [&] { kernel->fp8_fn(n, n, n, alpha, Aq, sa, Bq, sb, 0.f, C); });
      ref_ms = time_ms([&] { lt.run(alpha, Aq, Bq, 0.f, Ref); });
      exact = exact_err(n, alpha, Aq, sa, Bq, sb, C);
      if (!sa)
        printf("# cublaslt,%d,max_rel_err vs exact %.2e\n", n,
               exact_err(n, alpha, Aq, sa, Bq, sb, Ref));
      for (void *p : {(void *)Aq, (void *)Bq, (void *)packed, (void *)sa,
                      (void *)sb, (void *)amax})
        CUDA_CHECK(cudaFree(p));
    } else {
      // FP16 copies of A and B, converted once outside the timed loop.
      std::vector<__half> hAh(count), hBh(count);
      for (size_t i = 0; i < count; ++i) hAh[i] = __float2half(hA[i]);
      for (size_t i = 0; i < count; ++i) hBh[i] = __float2half(hB[i]);
      __half *Ah, *Bh;
      CUDA_CHECK(cudaMalloc(&Ah, count * sizeof(__half)));
      CUDA_CHECK(cudaMalloc(&Bh, count * sizeof(__half)));
      CUDA_CHECK(cudaMemcpy(Ah, hAh.data(), count * sizeof(__half),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(Bh, hBh.data(), count * sizeof(__half),
                            cudaMemcpyHostToDevice));
      ms = time_ms([&] { kernel->half_fn(n, n, n, 1.f, Ah, Bh, 0.f, C); });
      ref_ms =
          time_ms([&] { gemm_cublas_half(n, n, n, 1.f, Ah, Bh, 0.f, Ref); });
      CUDA_CHECK(cudaFree(Ah));
      CUDA_CHECK(cudaFree(Bh));
    }
    CUDA_CHECK(cudaMemcpy(hC.data(), C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRef.data(), Ref, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0, max_ref = 0;
    for (size_t i = 0; i < count; ++i) {
      max_err = fmax(max_err, fabs(double(hC[i]) - hRef[i]));
      max_ref = fmax(max_ref, fabs(double(hRef[i])));
    }
    double rel = exact >= 0 ? exact : max_err / max_ref;
    double flops = 2.0 * n * n * n;
    double gflops = flops / (ms * 1e6), ref_gflops = flops / (ref_ms * 1e6);
    printf("%s,%d,%.4f,%.1f,%.1f,%.1f,%.2e\n", argv[1], n, ms, gflops,
           ref_gflops, 100.0 * gflops / ref_gflops, rel);
    if (!(rel <= tolerance)) ok = false;

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(Ref));
  }
  CUBLAS_CHECK(cublasDestroy(g_handle));
  if (!ok) fprintf(stderr, "FAIL: error above %.0e\n", tolerance);
  return ok ? 0 : 1;
}
