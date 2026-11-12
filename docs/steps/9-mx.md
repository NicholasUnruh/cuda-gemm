# Step 9: MX block scaling (MXFP8, MXFP4, NVFP4)

Source: [src/fp8.cu](../../src/fp8.cu) (`mxfp8_kernel`), [src/quant.cu](../../src/quant.cu). RTX 4060, 4096³: MXFP8 37,856 GFLOPS, 65.5% of per-tensor cuBLASLt FP8 and of step 8.

## What changed

OCP MX gives every 32 elements along K a power-of-two scale (E8M0); elements are E4M3 (MXFP8) or E2M1 (MXFP4). chip-lab ranks a 32-term MXFP8 unit best per mm². What does MX cost on a GPU without MX hardware?

One `m16n8k32` MMA covers exactly one block of A and of B. Each MMA's partial sum `p` comes from a fresh accumulator and is added into FP32 as `p · 2^(sa + sb − 254)`. Step 8 adds once per 64-deep K tile.

## Cost of scaling

| Kernel, 4096³ | TFLOP/s | FP32 work per MMA |
|---|---:|---|
| `fp8`, promote per 64-deep tile (step 8) | 57.8 | 2 FADD |
| `fp8`, promote per MMA ([ablation](../results/fp8-promotion.csv)) | 47.6 | 4 FADD |
| `mxfp8` | 37.9 | 4 FMUL + 4 FFMA |

Per-MMA promotion costs half; scale multiplies the rest. The first build needed 182 registers, one block per SM. `__launch_bounds__(256, 2)` caps it at 128 (148 bytes spilled) for two blocks, 8.9% faster ([ablation](../results/mxfp8-occupancy.csv)). ncu: tensor pipe 30% busy (step 8: 45%), 310 M instructions against 109 M, long-scoreboard stalls on top ([SASS](../results/mxfp8-sass.txt): 32 QMMA, 248 FMUL, 192 FFMA).

## Accuracy

Median / p99 error on chip-lab's vectors, K = 4096, Gaussian ([all distributions](../results/numerics-rtx4060.csv), [chip-lab](../results/chiplab-gemm-error.csv)):

| Format | GPU | chip-lab model |
|---|---|---|
| FP8, per tensor | 6.22e-4 / 2.32e-3 | 6.2e-4 / 2.3e-3 |
| MXFP8 | 6.94e-4 / 2.62e-3 | 7.0e-4 / 2.6e-3 |
| MXFP4 | 2.60e-3 / 1.05e-2 | 2.6e-3 / 1.1e-2 |
| NVFP4 | 2.26e-3 / 8.14e-3 | not modelled |

The GPU matches chip-lab's bit-exact models on all four distributions. MXFP8 is slightly worse than per-tensor FP8, as chip-lab also finds: E4M3's exponent already covers this data, and MX's power-of-two rule wastes headroom.

## FP4 without FP4 tensor cores

- **MXFP4** codes are re-encoded exactly as E4M3 and run through `mxfp8_kernel` at the MXFP8 rate. Error against the exact reference is 0, as step 8's probe predicts: E2M1 products are multiples of 2⁻³ up to 36, so 32-term sums fit the adder's 14 bits.
- **NVFP4** (E4M3 scale per 16) cannot use a k32 MMA, which would span two scales. Scaled elements (6 significant bits, 2⁻¹⁰..2688) are exact in FP16, so the emulation decodes and runs cuBLAS FP16.

## Cost of quantizing

Both 4096² operands, as a share of the format's GEMM ([cost block](../results/numerics-rtx4060.csv), one run): MXFP8 0.67 ms (19%). Per-tensor FP8 1.30 ms (48%), since it needs an amax pass. MXFP4 0.89 ms. NVFP4 4.95 ms (116%) with its FP16 decode: more accurate than MXFP4, but on Ada the emulation costs more than the GEMM.

## Verification

Quantizers match chip-lab's C++ byte for byte on 27 M elements ([cross-check](../results/quant-crosscheck.csv)); `bench mxfp8` and `bench mxfp4` check every output against exact FP64 dot products. [Journal](../journal/2026-11-12-mx.md).
