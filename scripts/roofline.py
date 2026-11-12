# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Roofline of every kernel from `make profile` output (Nsight Compute raw CSV).

Usage: uv run scripts/roofline.py <ncu.csv> <out.png|.webp> <gpu name> [light|dark]
Every number comes from counters: achieved FLOPs, time and DRAM bytes from
each kernel's own launch, peaks from the same run's peak_sustained metrics.
"""

import csv
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import THEMES

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


# Label offset (points) and alignment per kernel, chosen so no label touches a
# point, another label or a ceiling: reg2d and vec sit 2% apart in intensity.
LABEL_POS = {
    "vec": ((0, 7), "center", "bottom"),
    "reg2d": ((0, -8), "center", "top"),
    "wmma": ((-8, 0), "right", "center"),
}


def main(src, dst, gpu, theme="light"):
    rows = load(src)
    r0 = rows[0]
    sm_hz = r0["sm__cycles_elapsed.avg.per_second"]
    peak_fp32 = 2 * r0[FP32 + ".peak_sustained"] * sm_hz
    peak_tensor = r0[TENSOR + ".peak_sustained"] * sm_hz
    peak_bw = (
        r0["dram__bytes.sum.peak_sustained"] * r0["dram__cycles_elapsed.avg.per_second"]
    )
    print(
        f"peak_fp32_gflops,{peak_fp32 / 1e9:.0f}\npeak_tensor_gflops,{peak_tensor / 1e9:.0f}"
        f"\npeak_dram_gbs,{peak_bw / 1e9:.0f}\nkernel,flop_per_byte,gflops"
    )
    for r in rows:
        r["flops"] = 2 * r[FP32] + r[TENSOR]
        r["bytes"] = r["dram__bytes_read.sum"] + r["dram__bytes_write.sum"]
        r["ai"] = r["flops"] / r["bytes"]
        r["perf"] = r["flops"] / (r["gpu__time_duration.sum"] * 1e-9)
        print(f"{r['name']},{r['ai']:.2f},{r['perf'] / 1e9:.1f}")

    c = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    lo = min(r["ai"] for r in rows) / 2
    hi = max(r["ai"] for r in rows) * 4
    ax.set_xlim(lo / 1.1, hi * 1.1)
    ax.set_ylim(min(r["perf"] for r in rows) / 1e9 / 2, peak_tensor / 1e9 * 2)
    xs = [lo * (hi / lo) ** (i / 200) for i in range(201)]
    # Tensor ceiling label above its line, FP32 label below its own, both at
    # the right edge where no kernel sits.
    for peak, label, va, nudge in [
        (peak_fp32, "FP32 FMA peak", "top", 1 / 1.12),
        (peak_tensor, "FP16 tensor peak", "bottom", 1.12),
    ]:
        ax.plot(
            xs, [min(peak, peak_bw * x) / 1e9 for x in xs], color=c["muted"], lw=1.5
        )
        ax.text(
            hi,
            peak / 1e9 * nudge,
            f"{label} {peak / 1e12:.1f} TFLOP/s",
            ha="right",
            va=va,
            fontsize=8,
            color=c["muted"],
        )
    for r in rows:
        ax.scatter(
            r["ai"],
            r["perf"] / 1e9,
            s=48,
            color=c["tc"] if r[TENSOR] > 0 else c["fp32"],
            edgecolor=c["surface"],
            linewidth=2,
            zorder=3,
        )
        offset, ha, va = LABEL_POS.get(r["name"], ((8, 0), "left", "center"))
        ax.annotate(
            r["name"],
            (r["ai"], r["perf"] / 1e9),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=8,
            color=c["ink"],
        )
    ax.scatter([], [], s=48, color=c["fp32"], label="FP32 kernels")
    ax.scatter([], [], s=48, color=c["tc"], label="FP16 tensor-core kernel")
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for t in leg.get_texts():
        t.set_color(c["ink"])
    ax.set_xlabel("Arithmetic intensity (FLOP per DRAM byte)", color=c["ink"])
    ax.set_ylabel("Achieved GFLOP/s", color=c["ink"])
    ax.grid(True, which="major", color=c["grid"], lw=0.6)
    for s in ax.spines.values():
        s.set_color(c["grid"])
    ax.tick_params(colors=c["muted"], labelsize=8, which="both")
    ax.set_title(
        f"{gpu}: roofline at 4096³ from Nsight Compute counters",
        fontsize=10,
        color=c["ink"],
        loc="left",
    )
    fig.tight_layout()
    # DRAM label just above the bandwidth slope, rotated to match it on screen.
    x0, x1 = lo * 1.3, lo * 2.6
    p0, p1 = (ax.transData.transform((x, peak_bw * x / 1e9)) for x in (x0, x1))
    angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(
        x0,
        peak_bw * x0 / 1e9 * 1.25,
        f"DRAM {peak_bw / 1e9:.0f} GB/s",
        rotation=angle,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        fontsize=8,
        color=c["muted"],
    )
    webp = {"quality": 82} if dst.endswith(".webp") else None
    fig.savefig(dst, facecolor=c["surface"], pil_kwargs=webp)


if __name__ == "__main__":
    main(*sys.argv[1:])
