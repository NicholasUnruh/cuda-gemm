# Step 9: MX block scaling (MXFP8, MXFP4, NVFP4)

Source: [src/fp8.cu](../../src/fp8.cu) (`mxfp8_kernel`), [src/quant.cu](../../src/quant.cu). RTX 4060, 4096³: MXFP8 MXFP8_GF GFLOPS, MXFP8_PCT% of per-tensor cuBLASLt FP8, MXFP8_VS_FP8% of step 8.

## What changed

OCP MX gives every 32 consecutive elements along K a shared power-of-two scale (E8M0, one byte), then stores the elements in E4M3 (MXFP8) or E2M1 (MXFP4). chip-lab's numerics study ranks a 32-term MXFP8 unit as its best training datapath per mm². The question here is what MX costs on a GPU without MX hardware.

One `m16n8k32` MMA covers exactly one 32-element block of A and of B. The kernel therefore takes each MMA's partial sum out of a fresh accumulator and adds it into FP32 as `acc += p · 2^(sa + sb − 254)`: one FMUL and one FFMA per output element per MMA. Step 8 adds once per 64-deep K tile instead. Each thread loads its row's two scale bytes one tile ahead from global memory, and they sit in shared memory beside the operands.

## Cost of scaling

| Kernel, 4096³ | TFLOP/s | FP32 work per MMA |
|---|---:|---|
| `fp8`, promotion per 64-deep tile (step 8) | 57.8 | 2 FADD |
| `fp8`, promotion after every MMA ([ablation](../results/fp8-promotion.csv)) | 47.6 | 4 FADD |
| `mxfp8` | MXFP8_TF | 4 FMUL + 4 FFMA |

Profile: PROFILE_SENTENCE

## Accuracy

Median / p99 error on chip-lab's vectors at K = 4096 ([results](../results/numerics-rtx4060.csv), [chip-lab](../results/chiplab-gemm-error.csv)):

ACC_TABLE

The GPU reproduces chip-lab's bit-exact models to two digits, so the formats behave identically in silicon and on the GPU. On these distributions MXFP8 is slightly less accurate than per-tensor FP8 (chip-lab finds the same ordering): E4M3's own exponent already covers the row-scaled data, so the block scale buys range that is not needed. MX's rule maps each block maximum into [256, 512), and values above 448 saturate.

## FP4 on a GPU without FP4 tensor cores

- **MXFP4** codes are re-encoded as the same values in E4M3 (`e2m1_to_e4m3`) and run through `mxfp8_kernel` with the MXFP4 scales. The bench row `mxfp4` therefore runs at the MXFP8 rate. Its error against the exact reference is 0. That fits the probe of step 8: E2M1 products are multiples of 2⁻³ no larger than 36, so a 32-term sum (under 2¹¹) loses nothing to an adder that keeps 13 bits below the leading bit.
- **NVFP4** (E2M1 elements, an E4M3 scale per 16, an FP32 scale per tensor) cannot use a k32 FP8 MMA, because one MMA would span two scales. Each element times its block scale has at most 6 significant bits within 2⁻¹⁰..2688, so it is exact in FP16. The emulation decodes to FP16 and runs cuBLAS FP16 GEMM.

## Cost of quantizing

QUANT_SENTENCE

## Verification

The quantizers match chip-lab's C++ references (and mx-training's NVFP4 contract) byte for byte on 27 million elements ([cross-check](../results/quant-crosscheck.csv)). `bench mxfp8` and `bench mxfp4` check every output against exact FP64 dot products of the codes with their scales (tolerance 1e-3). Process notes: [journal](../journal/2026-11-12-mx.md).
