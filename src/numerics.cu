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
  void *qa, *qb;            // GEMM operands (largest format: 16-bit)
  uint8_t *pa, *pb;         // packed FP4 codes
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

// One format: `quantize` turns b.A and b.Bt into GEMM operands (and scales);
// `gemm` writes b.C, given the factor the output needs. Formats with a
// per-tensor scale set tensor_max: codes are x * tensor_max / amax, so the
// factor is (amax_a / tensor_max) * (amax_b / tensor_max).
struct Path {
  const char *format, *impl;
  bool fp8_tensor_cores;
  double tensor_max;
  std::function<void(Buffers &)> quantize;
  std::function<void(Buffers &, float)> gemm;
};

uint8_t *u8(void *p) { return static_cast<uint8_t *>(p); }

// cuBLAS GemmEx in the TN layout of sgemm_tn, 16-bit inputs, FP32 accumulate.
void gemm16_tn(const Buffers &b, float alpha, cudaDataType type) {
  float zero = 0.f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, b.rows, b.rows,
                            b.K, &alpha, b.qb, type, b.K, b.qa, type, b.K,
                            &zero, b.C, CUDA_R_32F, b.rows, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

std::vector<Path> paths() {
  auto n_of = [](const Buffers &b) { return size_t(b.rows) * b.K; };
  auto e4m3 = [=](Buffers &b) {
    quant_amax(b.A, n_of(b), b.amax);
    quant_amax(b.Bt, n_of(b), b.amax + 1);
    quant_e4m3_tensor(b.A, n_of(b), b.amax, u8(b.qa));
    quant_e4m3_tensor(b.Bt, n_of(b), b.amax + 1, u8(b.qb));
  };
  auto mxfp8 = [=](Buffers &b) {
    quant_mx(b.A, n_of(b), false, u8(b.qa), b.sa);
    quant_mx(b.Bt, n_of(b), false, u8(b.qb), b.sb);
  };
  auto mxfp4 = [=](Buffers &b) {
    quant_mx(b.A, n_of(b), true, b.pa, b.sa);
    quant_mx(b.Bt, n_of(b), true, b.pb, b.sb);
    e2m1_to_e4m3(b.pa, n_of(b), u8(b.qa));
    e2m1_to_e4m3(b.pb, n_of(b), u8(b.qb));
  };
  auto nvfp4 = [=](Buffers &b) {
    quant_amax(b.A, n_of(b), b.amax);
    quant_amax(b.Bt, n_of(b), b.amax + 1);
    quant_nvfp4(b.A, n_of(b), b.amax, b.pa, b.sa);
    quant_nvfp4(b.Bt, n_of(b), b.amax + 1, b.pb, b.sb);
    nvfp4_to_half(b.pa, b.sa, n_of(b), static_cast<__half *>(b.qa));
    nvfp4_to_half(b.pb, b.sb, n_of(b), static_cast<__half *>(b.qb));
  };
  auto mine = [](GemmFp8Fn fn, bool mx) {
    return [=](Buffers &b, float alpha) {
      fn(b.rows, b.rows, b.K, alpha, u8(b.qa), mx ? b.sa : nullptr, u8(b.qb),
         mx ? b.sb : nullptr, 0.f, b.C);
    };
  };
  return {
      {"fp32", "cuBLAS", false, 0, [](Buffers &) {},
       [](Buffers &b, float) { sgemm_tn(b.rows, b.rows, b.K, b.A, b.Bt, b.C); }},
      {"bf16", "cuBLAS", false, 0,
       [=](Buffers &b) {
         quant_bf16(b.A, n_of(b), static_cast<__nv_bfloat16 *>(b.qa));
         quant_bf16(b.Bt, n_of(b), static_cast<__nv_bfloat16 *>(b.qb));
       },
       [](Buffers &b, float alpha) { gemm16_tn(b, alpha, CUDA_R_16BF); }},
      {"e4m3_tensor", "fp8 (step 8)", true, 448, e4m3, mine(gemm_fp8, false)},
      {"e4m3_tensor", "cuBLASLt", true, 448, e4m3,
       [](Buffers &b, float alpha) {
         static LtFp8 *lt = nullptr;
         static int lt_rows = 0, lt_k = 0;
         if (!lt || lt_rows != b.rows || lt_k != b.K) {
           delete lt;
           lt = new LtFp8(b.rows, b.rows, b.K);
           lt_rows = b.rows, lt_k = b.K;
         }
         lt->run(alpha, u8(b.qa), u8(b.qb), 0.f, b.C);
       }},
      {"mxfp8", "mxfp8 (step 9)", true, 0, mxfp8, mine(gemm_mxfp8, true)},
      {"mxfp4", "mxfp8 kernel, E2M1 values", true, 0, mxfp4,
       mine(gemm_mxfp8, true)},
      {"nvfp4", "FP16 cuBLAS on decoded values", false, 6.0 * 448.0, nvfp4,
       [](Buffers &b, float alpha) { gemm16_tn(b, alpha, CUDA_R_16F); }},
  };
}

// The factor C needs, once the quantizer has run.
float output_factor(const Path &p, Buffers &b) {
  if (p.tensor_max == 0) return 1.f;
  return float(double(amax_of(b.amax)) / p.tensor_max * amax_of(b.amax + 1) /
               p.tensor_max);
}

Buffers alloc(int rows, int K) {
  Buffers b{rows, K};
  size_t n = size_t(rows) * K;
  CUDA_CHECK(cudaMalloc(&b.A, n * 4));
  CUDA_CHECK(cudaMalloc(&b.Bt, n * 4));
  CUDA_CHECK(cudaMalloc(&b.C, size_t(rows) * rows * 4));
  CUDA_CHECK(cudaMalloc(&b.qa, n * 2));
  CUDA_CHECK(cudaMalloc(&b.qb, n * 2));
  CUDA_CHECK(cudaMalloc(&b.pa, n / 2));
  CUDA_CHECK(cudaMalloc(&b.pb, n / 2));
  CUDA_CHECK(cudaMalloc(&b.sa, n / 16));
  CUDA_CHECK(cudaMalloc(&b.sb, n / 16));
  CUDA_CHECK(cudaMalloc(&b.amax, 8));
  return b;
}

void release(Buffers &b) {
  for (void *p : {(void *)b.A, (void *)b.Bt, (void *)b.C, b.qa, b.qb,
                  (void *)b.pa, (void *)b.pb, (void *)b.sa, (void *)b.sb,
                  (void *)b.amax})
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
        p.quantize(b);
        p.gemm(b, output_factor(p, b));
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
    p.quantize(b);
    float alpha = output_factor(p, b);
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
