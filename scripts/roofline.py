# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Roofline of every kernel from `make profile` output (Nsight Compute raw CSV).

Usage: uv run scripts/roofline.py <ncu.csv> <out.png> <gpu name>
Every number comes from counters: achieved FLOPs, time and DRAM bytes from
each kernel's own launch, peaks from the same run's peak_sustained metrics.
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FP32 = "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
TENSOR = "sm__ops_path_tensor_src_fp16_dst_fp32.sum"


def load(path):
    """Kernel rows of the ncu raw CSV pages, numbers parsed."""
    with open(path) as f:
        lines = [l for l in f if l.startswith('"')]  # drop bench's own stdout
    rows = []
    for r in csv.DictReader(lines):
        if r["ID"] in ("ID", ""):  # repeated header or units row
            continue
        rec = {"name": r["Kernel Name"].split("_kernel")[0]}
        for k, v in r.items():
            try:
                rec[k] = float(v.replace(",", ""))
            except (ValueError, AttributeError):
                pass
        rows.append(rec)
    return rows


def main(src, dst, gpu):
    rows = load(src)
    r0 = rows[0]
    sm_hz = r0["sm__cycles_elapsed.avg.per_second"]
    peak_fp32 = 2 * r0[FP32 + ".peak_sustained"] * sm_hz
    peak_tensor = r0[TENSOR + ".peak_sustained"] * sm_hz
    peak_bw = r0["dram__bytes.sum.peak_sustained"] * r0["dram__cycles_elapsed.avg.per_second"]
    print(f"peak_fp32_gflops,{peak_fp32 / 1e9:.0f}\npeak_tensor_gflops,{peak_tensor / 1e9:.0f}"
          f"\npeak_dram_gbs,{peak_bw / 1e9:.0f}\nkernel,flop_per_byte,gflops")
    for r in rows:
        r["flops"] = 2 * r[FP32] + r[TENSOR]
        r["bytes"] = r["dram__bytes_read.sum"] + r["dram__bytes_write.sum"]
        r["ai"] = r["flops"] / r["bytes"]
        r["perf"] = r["flops"] / (r["gpu__time_duration.sum"] * 1e-9)
        print(f"{r['name']},{r['ai']:.2f},{r['perf'] / 1e9:.1f}")

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fp32_c, tc_c = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    lo = min(r["ai"] for r in rows) / 2
    hi = max(r["ai"] for r in rows) * 4
    xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    for peak, label in [(peak_fp32, "FP32 FMA peak"), (peak_tensor, "FP16 tensor peak")]:
        ax.plot(xs, [min(peak, peak_bw * x) / 1e9 for x in xs], color=muted, lw=1.5)
        ax.text(hi, peak / 1e9 * 1.1, f"{label} {peak / 1e12:.1f} TFLOP/s",
                ha="right", va="bottom", fontsize=8, color=muted)
    ax.text(lo * 1.2, peak_bw * lo * 1.2 / 1e9 * 1.4, f"DRAM {peak_bw / 1e9:.0f} GB/s",
            fontsize=8, color=muted)
    for r in rows:
        tc = r[TENSOR] > 0
        ax.scatter(r["ai"], r["perf"] / 1e9, s=48, color=tc_c if tc else fp32_c,
                   edgecolor=surface, linewidth=2, zorder=3)
        ax.annotate(r["name"], (r["ai"], r["perf"] / 1e9), xytext=(7, -3),
                    textcoords="offset points", fontsize=8, color=ink)
    ax.scatter([], [], s=48, color=fp32_c, label="FP32 kernels")
    ax.scatter([], [], s=48, color=tc_c, label="FP16 tensor-core kernel")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Arithmetic intensity (FLOP per DRAM byte)", color=ink)
    ax.set_ylabel("Achieved GFLOP/s", color=ink)
    ax.grid(True, which="major", color=grid, lw=0.6)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    ax.set_title(f"{gpu}: roofline at 4096³ from Nsight Compute counters",
                 fontsize=10, color=ink, loc="left")
    fig.tight_layout()
    fig.savefig(dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
