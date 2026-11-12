// Accuracy and cost of each number format on this GPU: `bench numerics`.
//
// Accuracy repeats the GEMM-error experiment of the chip-lab numerics study
// (openroad-chip-lab tools/numerics/gemm_error.cpp) on the same vectors: the
// same generator, seeds and draw order give 2,000 pairs (a_s, b_s) per
// distribution and K. Here they are the rows of A and Bt (padded with zero
// rows to 2048), one GEMM computes every a_s . b_t, and the diagonal is
// scored like the study: |y - y64| / sum |a_i b_i| against the FP64 dot
// product of the original data. The GPU sees the data rounded to FP32.
//
// Cost times quantizing both 4096 x 4096 operands and the 4096^3 GEMM.
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "gemm.cuh"
#include "quant.cuh"
#include "util.cuh"

namespace {

cublasHandle_t handle;

// Device buffers for one format path: inputs as floats (rows x K each), the
// quantized copies and C (rows x rows).
struct Buffers {
  int rows, K;
  float *A, *Bt, *C;
  void *qa, *qb;            // quantized operands (largest format: BF16)
  uint8_t *sa, *sb;         // block scales
  unsigned *amax;           // two tensor amaxes
};

// C = A Bt^T for row-major A (M x K) and Bt (N x K): column-major cuBLAS
// computes C^T = Bt A^T, with Bt read transposed.
void sgemm_tn(int M, int N, int K, const float *A, const float *Bt, float *C) {
  float one = 1.f, zero = 0.f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, Bt,
                           K, A, K, &zero, C, N));
}

float amax_of(const unsigned *d) {
  unsigned bits;
  CUDA_CHECK(cudaMemcpy(&bits, d, 4, cudaMemcpyDeviceToHost));
  float f;
  memcpy(&f, &bits, 4);
  return f > 0.f ? f : 1.f;
}

// One format: `quantize` turns b.A and b.Bt into b.qa/b.qb (+ scales) and
// returns the factor the GEMM output needs; `gemm` writes b.C.
struct Path {
  const char *format, *impl;
  bool fp8_tensor_cores;
  std::function<float(Buffers &)> quantize;
  std::function<void(Buffers &, float)> gemm;
};

std::vector<Path> paths() {
  auto none = [](Buffers &) { return 1.f; };
  auto bf16 = [](Buffers &b) {
    size_t n = size_t(b.rows) * b.K;
    quant_bf16(b.A, n, static_cast<__nv_bfloat16 *>(b.qa));
    quant_bf16(b.Bt, n, static_cast<__nv_bfloat16 *>(b.qb));
    return 1.f;
  };
  auto e4m3 = [](Buffers &b) {
    size_t n = size_t(b.rows) * b.K;
    quant_amax(b.A, n, b.amax);
    quant_amax(b.Bt, n, b.amax + 1);
    quant_e4m3_tensor(b.A, n, b.amax, static_cast<uint8_t *>(b.qa));
    quant_e4m3_tensor(b.Bt, n, b.amax + 1, static_cast<uint8_t *>(b.qb));
    return -1.f;  // the factor needs the amaxes: read after the timing
  };
  return {
      {"fp32", "cuBLAS", false, none,
       [](Buffers &b, float) { sgemm_tn(b.rows, b.rows, b.K, b.A, b.Bt, b.C); }},
      {"bf16", "cuBLAS", false, bf16,
       [](Buffers &b, float) {
         float one = 1.f, zero = 0.f;
         CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, b.rows,
                                   b.rows, b.K, &one, b.qb, CUDA_R_16BF, b.K,
                                   b.qa, CUDA_R_16BF, b.K, &zero, b.C,
                                   CUDA_R_32F, b.rows, CUBLAS_COMPUTE_32F,
                                   CUBLAS_GEMM_DEFAULT));
       }},
      {"e4m3_tensor", "fp8 (step 8)", true, e4m3,
       [](Buffers &b, float alpha) {
         gemm_fp8(b.rows, b.rows, b.K, alpha, static_cast<uint8_t *>(b.qa),
                  static_cast<uint8_t *>(b.qb), 0.f, b.C);
       }},
      {"e4m3_tensor", "cuBLASLt", true, e4m3,
       [](Buffers &b, float alpha) {
         static LtFp8 *lt = nullptr;
         static int lt_rows = 0, lt_k = 0;
         if (!lt || lt_rows != b.rows || lt_k != b.K) {
           delete lt;
           lt = new LtFp8(b.rows, b.rows, b.K);
           lt_rows = b.rows, lt_k = b.K;
         }
         lt->run(alpha, static_cast<uint8_t *>(b.qa),
                 static_cast<uint8_t *>(b.qb), 0.f, b.C);
       }},
  };
}

// The factor C needs after the GEMM, once the quantizer has run.
float output_factor(Buffers &b, float q) {
  if (q >= 0.f) return q;
  // E4M3 per tensor: codes are x * 448 / amax.
  return float(double(amax_of(b.amax)) / 448.0 * amax_of(b.amax + 1) / 448.0);
}

Buffers alloc(int rows, int K) {
  Buffers b{rows, K};
  size_t n = size_t(rows) * K;
  CUDA_CHECK(cudaMalloc(&b.A, n * 4));
  CUDA_CHECK(cudaMalloc(&b.Bt, n * 4));
  CUDA_CHECK(cudaMalloc(&b.C, size_t(rows) * rows * 4));
  CUDA_CHECK(cudaMalloc(&b.qa, n * 2));
  CUDA_CHECK(cudaMalloc(&b.qb, n * 2));
  CUDA_CHECK(cudaMalloc(&b.sa, n / 16));
  CUDA_CHECK(cudaMalloc(&b.sb, n / 16));
  CUDA_CHECK(cudaMalloc(&b.amax, 8));
  return b;
}

void release(Buffers &b) {
  for (void *p : {(void *)b.A, (void *)b.Bt, (void *)b.C, b.qa, b.qb,
                  (void *)b.sa, (void *)b.sb, (void *)b.amax})
    CUDA_CHECK(cudaFree(p));
}

void accuracy(bool fp8_ok) {
  const unsigned samples = 2000;
  const int rows = 2048;
  printf("distribution,K,format,impl,median_err,p99_err\n");
  for (const std::string dist : {"gaussian", "laplace", "outliers", "rows"})
    for (unsigned K : {256u, 4096u}) {
      // Verbatim from gemm_error.cpp, so the vectors are identical.
      std::mt19937_64 rng(K * 7919 + dist.size());
      std::normal_distribution<double> normal(0, 1);
      std::exponential_distribution<double> expo(1);
      std::vector<std::vector<double>> as(samples, std::vector<double>(K)),
          bs(samples, std::vector<double>(K));
      for (unsigned s = 0; s < samples; ++s) {
        double row_scale = dist == "rows" ? std::exp(1.5 * normal(rng)) : 1;
        for (unsigned i = 0; i < K; ++i) {
          if (dist == "laplace") {
            as[s][i] = (rng() & 1 ? 1 : -1) * expo(rng);
            bs[s][i] = (rng() & 1 ? 1 : -1) * expo(rng);
          } else {
            as[s][i] = normal(rng) * row_scale;
            bs[s][i] = normal(rng);
            if (dist == "outliers" && rng() % 100 == 0) as[s][i] *= 100;
          }
        }
      }
      std::vector<double> exact(samples), magnitude(samples);
      std::vector<float> ha(size_t(rows) * K, 0.f), hb(size_t(rows) * K, 0.f);
      for (unsigned s = 0; s < samples; ++s)
        for (unsigned i = 0; i < K; ++i) {
          exact[s] += as[s][i] * bs[s][i];
          magnitude[s] += std::fabs(as[s][i] * bs[s][i]);
          ha[size_t(s) * K + i] = float(as[s][i]);
          hb[size_t(s) * K + i] = float(bs[s][i]);
        }
      Buffers b = alloc(rows, K);
      CUDA_CHECK(cudaMemcpy(b.A, ha.data(), ha.size() * 4,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(b.Bt, hb.data(), hb.size() * 4,
                            cudaMemcpyHostToDevice));
      std::vector<float> c(size_t(rows) * rows);
      for (const Path &p : paths()) {
        if (p.fp8_tensor_cores && !fp8_ok) continue;
        float q = p.quantize(b);
        p.gemm(b, output_factor(b, q));
        CUDA_CHECK(cudaMemcpy(c.data(), b.C, c.size() * 4,
                              cudaMemcpyDeviceToHost));
        std::vector<double> err(samples);
        for (unsigned s = 0; s < samples; ++s)
          err[s] = std::fabs(c[size_t(s) * rows + s] - exact[s]) / magnitude[s];
        std::sort(err.begin(), err.end());
        printf("%s,%u,%s,%s,%.2e,%.2e\n", dist.c_str(), K, p.format, p.impl,
               err[samples / 2], err[samples * 99 / 100]);
      }
      release(b);
    }
}

void cost(bool fp8_ok) {
  const int n = 4096;
  Buffers b = alloc(n, n);
  std::vector<float> h(size_t(n) * n);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (float &x : h) x = dist(rng);
  CUDA_CHECK(cudaMemcpy(b.A, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  for (float &x : h) x = dist(rng);
  CUDA_CHECK(cudaMemcpy(b.Bt, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  printf("format,impl,quantize_ms,gemm_ms,gemm_tflops,quantize_pct_of_gemm\n");
  for (const Path &p : paths()) {
    if (p.fp8_tensor_cores && !fp8_ok) continue;
    float qms = time_ms([&] { p.quantize(b); });
    float alpha = output_factor(b, p.quantize(b));
    float gms = time_ms([&] { p.gemm(b, alpha); });
    printf("%s,%s,%.4f,%.4f,%.2f,%.1f\n", p.format, p.impl, qms, gms,
           2.0 * n * n * n / (gms * 1e9), 100.0 * qms / gms);
  }
  release(b);
}

}  // namespace

int run_numerics() {
  CUBLAS_CHECK(cublasCreate(&handle));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  bool fp8_ok = prop.major * 10 + prop.minor >= 89;
  printf("# %s\n", prop.name);
  accuracy(fp8_ok);
  cost(fp8_ok);
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
