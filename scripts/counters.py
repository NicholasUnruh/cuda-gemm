# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Small multiples of the Nsight Compute counters that explain each step.

Usage: uv run scripts/counters.py <ncu.csv> <out.png|.webp> [light|dark] [counters.json]
Reads `make profile` output (docs/results/ncu-rtx4060.csv for the RTX 4060).
One panel per counter, one bar per kernel. The optional JSON holds every
plotted value plus the raw counters it came from, under their ncu names.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import THEMES
from roofline import FP8, TENSOR, load

REQ = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
SECT = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"


def stall(reason):
    return f"smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio"


# (id, title, unit, counters, value from a row). Stall ratios are warps stalled
# for that reason per cycle with an instruction issued.
PANELS = [
    (
        "dram_read",
        "DRAM read",
        "GB",
        ["dram__bytes_read.sum"],
        lambda r: r["dram__bytes_read.sum"] / 1e9,
    ),
    (
        "dram_write",
        "DRAM write",
        "MB",
        ["dram__bytes_write.sum"],
        lambda r: r["dram__bytes_write.sum"] / 1e6,
    ),
    ("gld_requests", "Global load requests", "millions", [REQ], lambda r: r[REQ] / 1e6),
    (
        "sectors_per_request",
        "Sectors per global load request",
        "32-bit: 4 = coalesced; 128-bit: 16",
        [SECT, REQ],
        lambda r: r[SECT] / r[REQ],
    ),
    (
        "fma_pipe",
        "FMA pipe busy",
        "% of peak",
        ["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"],
        lambda r: r["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"],
    ),
    (
        "tensor_pipe",
        "Tensor pipe busy",
        "% of peak",
        ["sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"],
        lambda r: r[
            "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
        ],
    ),
    (
        "warps_active",
        "Achieved occupancy",
        "% of 48 warps per SM",
        ["sm__warps_active.avg.pct_of_peak_sustained_active"],
        lambda r: r["sm__warps_active.avg.pct_of_peak_sustained_active"],
    ),
] + [
    (
        f"stall_{s}",
        f"Stall: {label}",
        "warps per issue cycle",
        [stall(s)],
        lambda r, s=s: r[stall(s)],
    )
    for s, label in [
        ("lg_throttle", "LG throttle (global queue full)"),
        ("long_scoreboard", "long scoreboard (global data)"),
        ("mio_throttle", "MIO throttle (shared-mem queue)"),
        ("barrier", "barrier (__syncthreads)"),
        ("math_pipe_throttle", "math pipe throttle"),
    ]
]

# Everything `make profile` collects, copied raw into the JSON.
RAW = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    SECT,
    REQ,
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "smsp__inst_executed.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    TENSOR,
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
] + [
    stall(s)
    for s in (
        "long_scoreboard",
        "mio_throttle",
        "short_scoreboard",
        "barrier",
        "math_pipe_throttle",
        "lg_throttle",
        "not_selected",
    )
]


def fmt(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:.3g}"


def main(src, dst, theme="light", json_out=None):
    rows = load(src)
    c = THEMES[theme]
    names = [r["name"] for r in rows]
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), dpi=133.34, sharey=True)
    fig.patch.set_facecolor(c["surface"])
    for ax, (_, title, unit, _, value) in zip(axes.flat, PANELS):
        vals = [value(r) for r in rows]
        ax.set_facecolor(c["surface"])
        colors = [c["tc"] if r[TENSOR] + r[FP8] > 0 else c["fp32"] for r in rows]
        ax.barh(range(len(rows)), vals, height=0.62, color=colors)
        top = max(vals) or 1
        ax.set_xlim(0, top * 1.45)
        for i, v in enumerate(vals):
            ax.text(v + top * 0.03, i, fmt(v), va="center", fontsize=8, color=c["ink"])
        ax.set_title(title, loc="left", fontsize=9.5, color=c["ink"], pad=14)
        ax.text(0, 1.02, unit, transform=ax.transAxes, fontsize=7.5, color=c["muted"])
        ax.set_xticks([])
        ax.set_yticks(range(len(rows)), names)
        ax.tick_params(colors=c["ink"], labelsize=8.5, length=0)
        for side, s in ax.spines.items():
            s.set_visible(side == "left")
            s.set_color(c["grid"])
    axes.flat[0].invert_yaxis()  # shared: step order top to bottom everywhere
    fig.suptitle(
        "Nsight Compute counters per kernel, RTX 4060, 4096³ (one profiled launch each)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=c["ink"],
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.2)
    webp = {"quality": 82} if dst.endswith(".webp") else None
    fig.savefig(dst, facecolor=c["surface"], pil_kwargs=webp)

    if json_out:
        out = {
            "source": src,
            "gpu": "RTX 4060 (sm_89)",
            "size": 4096,
            "kernels": names,
            "panels": [
                {
                    "id": pid,
                    "title": title,
                    "unit": unit,
                    "counters": ctrs,
                    "values": [round(value(r), 4) for r in rows],
                }
                for pid, title, unit, ctrs, value in PANELS
            ],
            "raw": {r["name"]: {k: r[k] for k in RAW} for r in rows},
        }
        with open(json_out, "w") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main(*sys.argv[1:])
