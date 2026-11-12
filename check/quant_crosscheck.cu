// Randomized cross-check: the GPU quantizers in src/quant.cu against the C++
// reference of the chip-lab numerics study (openroad-chip-lab,
// tools/numerics/datapaths.h), byte for byte, codes and scales.
// Build and run: make crosscheck CHIPLAB=<path to openroad-chip-lab>
//
// References: per-tensor E4M3 is dot8_chain's quantization (to_small of
// x * 448 / amax); MX is mx_chain's shared-exponent rule, copied below, with
// to_small for the elements. NVFP4 is not in the numerics study: its
// reference is mx-training's (tools/numerics/training/xcheck.cpp, copied in
// ref_nvfp4), again with to_small doing the rounding.
#include <cuda_fp8.h>

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../src/quant.cuh"
#include "tools/numerics/datapaths.h"

using Bytes = std::vector<uint8_t>;

static Bytes ref_e4m3_tensor(const std::vector<float> &x) {
  Vec d(x.begin(), x.end());
  double sa = 448.0 / amax(d);
  Bytes out(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = to_small(d[i] * sa, 4, 3, 448);
  return out;
}

// mx_chain's quantization of one operand (datapaths.h), block by block.
static void ref_mx(const std::vector<float> &x, bool fp4, Bytes &codes,
                   Bytes &scales) {
  int emax = fp4 ? 2 : 8;
  double max_value = fp4 ? 6.0 : 448.0;
  int exp_bits = fp4 ? 2 : 4, man_bits = fp4 ? 1 : 3;
  codes.assign(fp4 ? x.size() / 2 : x.size(), 0);
  scales.assign(x.size() / 32, 0);
  for (size_t i0 = 0; i0 < x.size(); i0 += 32) {
    double m = 0;
    for (size_t k = 0; k < 32; ++k) m = std::max(m, std::fabs(double(x[i0 + k])));
    int shared = m > 0 ? int(std::floor(std::log2(m))) - emax : -127;
    shared = std::clamp(shared, -127, 127);
    scales[i0 / 32] = uint8_t(shared + 127);
    for (size_t k = 0; k < 32; ++k) {
      uint8_t c = to_small(std::ldexp(double(x[i0 + k]), -shared), exp_bits,
                           man_bits, max_value);
      if (fp4)
        codes[(i0 + k) / 2] |= uint8_t(c << ((i0 + k) % 2 * 4));
      else
        codes[i0 + k] = c;
    }
  }
}

static float decode_e4m3(uint8_t c) {
  return __half2float(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3));
}

// The quantize() of mx-training's xcheck.cpp (openroad-chip-lab,
// tools/numerics/training), NVFP4 branch, in double throughout.
static void ref_nvfp4(const std::vector<float> &x, Bytes &codes, Bytes &scales) {
  double tensor_amax = 0;
  for (float v : x) tensor_amax = std::max(tensor_amax, std::fabs(double(v)));
  const double encode = 6.0 * 448.0 / (tensor_amax > 0 ? tensor_amax : 1.0);
  codes.assign(x.size() / 2, 0);
  scales.assign(x.size() / 16, 0);
  for (size_t i0 = 0; i0 < x.size(); i0 += 16) {
    double m = 0;
    for (size_t k = 0; k < 16; ++k)
      m = std::max(m, std::fabs(double(x[i0 + k])));
    uint8_t s = to_small(m / 6.0 * encode, 4, 3, 448.0);
    scales[i0 / 16] = s;
    double divide = decode_e4m3(s);
    for (size_t k = 0; k < 16; ++k) {
      uint8_t c = divide == 0 ? 0
                              : to_small(double(x[i0 + k]) * encode / divide,
                                         2, 1, 6.0);
      codes[(i0 + k) / 2] |= uint8_t(c << ((i0 + k) % 2 * 4));
    }
  }
}

// Data sets. "study": the four distributions of chip-lab's gemm_error at
// K = 4096, plus "sparse" (90% zeros). "wide": random bit patterns over every finite float, subnormals
// included. "ties": blocks whose values sit exactly halfway between two
// element codes after MX scaling, to exercise round-to-nearest-even.
static std::vector<float> make_data(const std::string &set, bool fp4,
                                    std::mt19937_64 &rng) {
  const size_t n = size_t(1) << 20;
  std::vector<float> x(n);
  if (set == "wide") {
    for (float &v : x) {
      uint32_t b;
      do b = uint32_t(rng());
      while ((b >> 23 & 0xff) == 0xff);
      memcpy(&v, &b, 4);
    }
  } else if (set == "ties") {
    // Positive values of the element format, then their midpoints.
    std::vector<double> vals;
    for (int c = 0; c < (fp4 ? 8 : 127); ++c)
      vals.push_back(fp4 ? std::array<double, 8>{0, .5, 1, 1.5, 2, 3, 4, 6}[c]
                         : decode_e4m3(uint8_t(c)));
    double top = vals.back();
    for (size_t i0 = 0; i0 < n; i0 += 32) {
      int r = int(rng() % 41) - 20;  // shared exponent of this block
      for (size_t k = 0; k < 32; ++k) {
        size_t c = rng() % (vals.size() - 1);
        double v = k == 0 ? top : (vals[c] + vals[c + 1]) / 2;
        x[i0 + k] = float(std::ldexp(rng() & 1 ? -v : v, r));
      }
    }
  } else {
    std::normal_distribution<double> normal(0, 1);
    std::exponential_distribution<double> expo(1);
    for (size_t r0 = 0; r0 < n; r0 += 4096) {
      double row_scale = set == "rows" ? std::exp(1.5 * normal(rng)) : 1;
      for (size_t i = 0; i < 4096; ++i) {
        double v;
        if (set == "laplace")
          v = (rng() & 1 ? 1 : -1) * expo(rng);
        else
          v = normal(rng) * row_scale;
        if (set == "outliers" && rng() % 100 == 0) v *= 100;
        if (set == "sparse" && rng() % 10 != 0) v = 0;
        x[r0 + i] = float(v);
      }
    }
  }
  return x;
}

static size_t mismatches(const Bytes &a, const Bytes &b) {
  size_t m = 0;
  for (size_t i = 0; i < a.size(); ++i) m += a[i] != b[i];
  return m;
}

int main() {
  float *x;
  uint8_t *codes, *scales;
  unsigned *am;
  const size_t n = size_t(1) << 20;
  cudaMalloc(&x, n * 4);
  cudaMalloc(&codes, n);
  cudaMalloc(&scales, n / 16);
  cudaMalloc(&am, 4);
  auto fetch = [&](uint8_t *p, size_t bytes) {
    Bytes h(bytes);
    if (cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "CUDA error\n");
      exit(2);
    }
    return h;
  };
  std::mt19937_64 rng(2026);
  size_t total = 0, bad = 0;
  printf("format,data,elements,code_mismatches,scale_mismatches\n");
  for (const char *fmt : {"e4m3_tensor", "mxfp8", "mxfp4", "nvfp4"})
    for (const char *set :
         {"gaussian", "laplace", "outliers", "rows", "sparse", "wide", "ties"}) {
      std::string f = fmt;
      bool fp4 = f == "mxfp4" || f == "nvfp4";
      if (std::string(set) == "ties" && (f == "e4m3_tensor" || f == "nvfp4"))
        continue;  // ties are built for power-of-two (MX) scales only
      std::vector<float> h = make_data(set, fp4, rng);
      cudaMemcpy(x, h.data(), n * 4, cudaMemcpyHostToDevice);
      Bytes rc, rs;
      size_t cb = fp4 ? n / 2 : n, sb = 0;
      if (f == "e4m3_tensor") {
        rc = ref_e4m3_tensor(h);
        quant_amax(x, n, am);
        quant_e4m3_tensor(x, n, am, codes);
      } else if (f == "nvfp4") {
        ref_nvfp4(h, rc, rs);
        sb = n / 16;
        quant_amax(x, n, am);
        quant_nvfp4(x, n, am, codes, scales);
      } else {
        ref_mx(h, fp4, rc, rs);
        sb = n / 32;
        quant_mx(x, n, fp4, codes, scales);
      }
      size_t mc = mismatches(fetch(codes, cb), rc);
      size_t ms = sb ? mismatches(fetch(scales, sb), rs) : 0;
      printf("%s,%s,%zu,%zu,%zu\n", fmt, set, n, mc, ms);
      total += n;
      bad += mc + ms;
    }
  printf("# %zu elements, %zu mismatched bytes\n", total, bad);
  return bad == 0 ? 0 : 1;
}
