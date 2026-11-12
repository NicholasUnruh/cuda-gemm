# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Key figure of the cp.async journal entry: GFLOPS and DRAM reads at 4096³ for
wmma, cpasync in row-major block order and cpasync in grouped order.

Usage: uv run scripts/journal_cpasync.py <out.png> [light|dark]
Reads docs/results/rtx4060.csv (medians) and docs/results/cpasync-order.csv
(row-major variant) and docs/results/ncu-rtx4060.csv (wmma DRAM reads).
"""

import csv
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import THEMES
from roofline import load


def median_gflops(kernel):
    with open("docs/results/rtx4060.csv") as f:
        rows = csv.DictReader(ln for ln in f if not ln.startswith("#"))
        return statistics.median(
            float(r["gflops"]) for r in rows if r["kernel"] == kernel and r["n"] == "4096"
        )


def main(dst, theme="light"):
    with open("docs/results/cpasync-order.csv") as f:
        order = {
            r["kernel_group"]: r
            for r in csv.DictReader(ln for ln in f if not ln.startswith("#"))
        }
    ncu = {r["name"]: r for r in load("docs/results/ncu-rtx4060.csv")}
    labels = ["wmma\n(step 6)", "cpasync\nrow-major order", "cpasync\ngroups of 8 rows"]
    gflops = [
        median_gflops("wmma"),
        float(order["1"]["gflops"]),
        median_gflops("cpasync"),
    ]
    dram = [
        ncu["wmma"]["dram__bytes_read.sum"] / 1e6,
        float(order["1"]["dram_read_bytes"]) / 1e6,
        float(order["8"]["dram_read_bytes"]) / 1e6,
    ]
    c = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    for ax, vals, title, fmt in [
        (axes[0], [g / 1e3 for g in gflops], "TFLOP/s (median of three)", "{:.1f}"),
        (axes[1], dram, "DRAM read per GEMM, MB (Nsight Compute)", "{:,.0f}"),
    ]:
        ax.set_facecolor(c["surface"])
        bars = ax.barh(labels[::-1], vals[::-1], color=c["tc"], height=0.6)
        for b, v in zip(bars, vals[::-1]):
            ax.text(
                b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                va="center", fontsize=8, color=c["ink"],
            )
        ax.set_title(title, fontsize=9, color=c["ink"], loc="left")
        ax.set_xlim(0, max(vals) * 1.25)
        ax.tick_params(colors=c["muted"], labelsize=8)
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[1].set_yticks([])
    fig.suptitle(
        "RTX 4060, FP16 GEMM 4096³: pipelining, then L2-friendly block order",
        fontsize=10, color=c["ink"], x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
