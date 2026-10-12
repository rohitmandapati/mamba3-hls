#!/usr/bin/env python3
"""
visualize_results.py
====================
Parses all csim sweep results under important_results/x86/ and produces:

  1. Four precision heatmaps (mean `out` error vs total_bits × int_bits):
       - pwl
       - cordic
       - pragma_pwl
       - pragma_cordic

  2. One grouped bar chart: max / mean / rmse for the SIX named configs,
     using only configs that PASSED (or the best FAIL if none passed).
     Configs: baseline, pragma, fixed+pwl (best pass), fixed+cordic (best pass),
              pragma+fixed+pwl (best pass), pragma+fixed+cordic (best pass)

All plots saved to: important_results/plots/
"""

import os
import re
import glob
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatterMathtext

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
X86_DIR    = os.path.join(SCRIPT_DIR, "x86")
FP_DIR     = os.path.join(X86_DIR, "fixed_pt_sweep")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_result(path):
    """Return dict with keys: passed, out_max, out_mean, out_rmse,
                               state_max, state_mean, state_rmse, time_ms
    Returns None if file is missing or unparseable."""
    if not os.path.exists(path):
        return None
    text = open(path).read()

    passed = "*** PASS ***" in text

    def _nums(pattern):
        m = re.search(pattern, text)
        if not m:
            return None, None, None
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    st_max, st_mean, st_rmse = _nums(
        r"state_out max / mean / rmse:\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)"
    )
    o_max, o_mean, o_rmse = _nums(
        r"(?<!state_)out\s+max / mean / rmse:\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)"
    )

    tm = re.search(r"([\d.]+)\s*msecs total time", text)
    time_ms = float(tm.group(1)) if tm else None

    return dict(
        passed=passed,
        out_max=o_max, out_mean=o_mean, out_rmse=o_rmse,
        state_max=st_max, state_mean=st_mean, state_rmse=st_rmse,
        time_ms=time_ms,
    )

# ---------------------------------------------------------------------------
# Load fixed-pt sweep grid for a given prefix (pwl / cordic / pragma_pwl / pragma_cordic)
# ---------------------------------------------------------------------------
TOTAL_BITS = [16, 20, 24, 28, 32]
INT_BITS   = [4, 6, 8, 10, 12, 16]

def load_fp_sweep(prefix):
    """Returns 2D array (total_bits rows × int_bits cols) of out_mean values.
    NaN where combination is invalid or file missing."""
    grid = np.full((len(TOTAL_BITS), len(INT_BITS)), np.nan)
    pass_grid = np.zeros((len(TOTAL_BITS), len(INT_BITS)), dtype=bool)

    for i, tb in enumerate(TOTAL_BITS):
        for j, ib in enumerate(INT_BITS):
            frac = tb - ib
            if ib >= tb or frac < 4:
                continue
            fname = os.path.join(FP_DIR, f"csim_{prefix}_fp{tb}i{ib}.txt")
            r = parse_result(fname)
            if r and r["out_mean"] is not None:
                grid[i, j] = r["out_mean"]
                pass_grid[i, j] = r["passed"]

    return grid, pass_grid

# ---------------------------------------------------------------------------
# Plot heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(prefix, title, ax):
    grid, pass_grid = load_fp_sweep(prefix)

    # Use log scale for colour — clip very small values
    log_grid = np.where(np.isnan(grid), np.nan, np.log10(np.clip(grid, 1e-9, None)))

    im = ax.imshow(log_grid, aspect="auto", origin="upper",
                   cmap="RdYlGn_r",
                   vmin=np.nanmin(log_grid) if not np.all(np.isnan(log_grid)) else -6,
                   vmax=np.nanmax(log_grid) if not np.all(np.isnan(log_grid)) else 0)

    ax.set_xticks(range(len(INT_BITS)))
    ax.set_xticklabels(INT_BITS, fontsize=8)
    ax.set_yticks(range(len(TOTAL_BITS)))
    ax.set_yticklabels(TOTAL_BITS, fontsize=8)
    ax.set_xlabel("Integer Bits", fontsize=9)
    ax.set_ylabel("Total Bits", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    # Annotate cells
    for i in range(len(TOTAL_BITS)):
        for j in range(len(INT_BITS)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="grey")
            else:
                passed = pass_grid[i, j]
                txt = f"{v:.2e}"
                color = "white" if log_grid[i, j] > (np.nanmin(log_grid) + (np.nanmax(log_grid) - np.nanmin(log_grid)) * 0.6) else "black"
                marker = "✓" if passed else "✗"
                ax.text(j, i, f"{marker}\n{txt}", ha="center", va="center",
                        fontsize=5.5, color=color, linespacing=1.4)

    return im

# ---------------------------------------------------------------------------
# Plot heatmaps — 4 in a 2×2 grid
# ---------------------------------------------------------------------------
SWEEP_CONFIGS = [
    ("pwl",           "Fixed + PWL"),
    ("cordic",        "Fixed + CORDIC"),
    ("pragma_pwl",    "Pragma + Fixed + PWL"),
    ("pragma_cordic", "Pragma + Fixed + CORDIC"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Fixed-Point Precision Sweep — Mean Output Error (log₁₀)\n"
             "✓=PASS  ✗=FAIL  (green=lower error)", fontsize=12, fontweight="bold", y=0.98)

ims = []
for ax, (prefix, title) in zip(axes.flat, SWEEP_CONFIGS):
    im = plot_heatmap(prefix, title, ax)
    ims.append(im)

# Shared colorbar
cbar = fig.colorbar(ims[0], ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
cbar.set_label("log₁₀(out mean error)", fontsize=9)

plt.savefig(os.path.join(PLOT_DIR, "heatmap_precision_sweep.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: heatmap_precision_sweep.png")

# ---------------------------------------------------------------------------
# Bar chart: best PASS per named config
# ---------------------------------------------------------------------------

def load_specific_fp(prefix, total_bits, int_bits):
    """Return the result dict for a specific fp config (e.g. fp24i10)."""
    fname = os.path.join(FP_DIR, f"csim_{prefix}_fp{total_bits}i{int_bits}.txt")
    r = parse_result(fname)
    if r:
        r["label_detail"] = f"fp{total_bits}i{int_bits}"
    return r

# Fixed-point configs use fp24i10 for both PWL and CORDIC
FP_TOTAL, FP_INT = 24, 10

BAR_CONFIGS = [
    ("Baseline",             os.path.join(X86_DIR, "csim_baseline.txt"), None,           None),
    ("Pragma",               os.path.join(X86_DIR, "csim_pragma.txt"),   None,           None),
    ("Fixed+PWL",            None,                                        "pwl",          (FP_TOTAL, FP_INT)),
    ("Fixed+CORDIC",         None,                                        "cordic",       (FP_TOTAL, FP_INT)),
    ("Pragma+Fixed\n+PWL",   None,                                        "pragma_pwl",   (FP_TOTAL, FP_INT)),
    ("Pragma+Fixed\n+CORDIC",None,                                        "pragma_cordic",(FP_TOTAL, FP_INT)),
]

results = []
labels  = []
for display_lbl, flat_path, sweep_prefix, fp_spec in BAR_CONFIGS:
    if flat_path is not None:
        r = parse_result(flat_path)
    else:
        tb, ib = fp_spec
        r = load_specific_fp(sweep_prefix, tb, ib)
    if r and r["out_mean"] is not None:
        results.append(r)
        detail = r.get("label_detail", "")
        labels.append(f"{display_lbl}\n({detail})" if detail else display_lbl)

if not results:
    print("No PASS results found for bar chart — skipping.")
else:
    x    = np.arange(len(results))
    w    = 0.22
    vals_max  = [r["out_max"]  for r in results]
    vals_mean = [r["out_mean"] for r in results]
    vals_rmse = [r["out_rmse"] for r in results]

    fig2, ax2 = plt.subplots(figsize=(max(9, 2.2 * len(results)), 5.5))
    ax2.bar(x - w, vals_max,  width=w, label="Max error",  color="#e74c3c", alpha=0.88)
    ax2.bar(x,     vals_mean, width=w, label="Mean error", color="#3498db", alpha=0.88)
    ax2.bar(x + w, vals_rmse, width=w, label="RMSE",       color="#2ecc71", alpha=0.88)

    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(LogFormatterMathtext())
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5, ha="center")
    ax2.set_ylabel("Output Error (log scale)", fontsize=10)
    ax2.set_title("Output Error Comparison (fp24i10 for fixed-point configs)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.35, linestyle="--")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bar_best_pass.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bar_best_pass.png")

# ===========================================================================
# SYNTHESIS PLOTS
# ===========================================================================

SYN_DIR  = os.path.join(SCRIPT_DIR, "syn")
SYN_CONFIGS = [
    ("baseline",            "Baseline"),
    ("pragma",              "Pragma"),
    ("fixed_pwl",           "Fixed+PWL"),
    ("fixed_cordic",        "Fixed+CORDIC"),
    ("pragma_fixed_pwl",    "Pragma+\nFixed+PWL"),
    ("pragma_fixed_cordic", "Pragma+\nFixed+CORDIC"),
]
COLORS6 = ["#3498db","#9b59b6","#2ecc71","#e67e22","#1abc9c","#e74c3c"]

def parse_syn_rpt(label):
    """Parse dut_csynth.rpt and return latency_ms, util dict, available dict."""
    path = os.path.join(SYN_DIR, f"syn_{label}.rpt")
    if not os.path.exists(path):
        return None
    text = open(path).read()

    # Latency: look for "Total" row in Latency Summary table
    lat_ms = None
    m = re.search(r"max\s*\|\s*\n.*?(\d+\.\d+)\s*ms", text)
    if not m:
        # try "X.XXX ms" format in the latency absolute column
        m = re.search(r"\|\s*(\d+\.\d+)\s*ms\s*\|\s*\1\s*ms\s*\|", text)
    if m:
        lat_ms = float(m.group(1))

    # Fallback: find the Latency (absolute) table - look for the Total row
    if lat_ms is None:
        m2 = re.search(
            r"Latency \(absolute\).*?\n.*?\|\s*[\d.]+\s*(ms|us)\s*\|\s*[\d.]+\s*(ms|us)\s*\|",
            text, re.DOTALL)
        if m2:
            nums = re.findall(r"\|\s*([\d.]+)\s*(ms|us)", m2.group(0))
            if nums:
                val, unit = nums[0]
                lat_ms = float(val) if unit == "ms" else float(val) / 1000.0

    # Utilization: grab the Total and Available rows
    util_m = re.search(
        r"\|Total\s*\|\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|",
        text)
    avail_m = re.search(
        r"\|Available\s*\|\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|\s*([\d-]+)\s*\|",
        text)

    util, avail = {}, {}
    if util_m:
        keys = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
        for k, v in zip(keys, util_m.groups()):
            util[k] = int(v) if v.strip() != "-" else 0
    if avail_m:
        keys = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
        for k, v in zip(keys, avail_m.groups()):
            avail[k] = int(v) if v.strip() != "-" else 1

    return {"lat_ms": lat_ms, "util": util, "avail": avail}


syn_data = {}
for lbl, _ in SYN_CONFIGS:
    d = parse_syn_rpt(lbl)
    if d:
        syn_data[lbl] = d

# ---------------------------------------------------------------------------
# Plot 3: Synthesis Latency Bar Chart
# ---------------------------------------------------------------------------
lat_labels = [disp for lbl, disp in SYN_CONFIGS if lbl in syn_data and syn_data[lbl]["lat_ms"] is not None]
lat_vals   = [syn_data[lbl]["lat_ms"] for lbl, _ in SYN_CONFIGS
              if lbl in syn_data and syn_data[lbl]["lat_ms"] is not None]
lat_colors = [COLORS6[i] for i, (lbl, _) in enumerate(SYN_CONFIGS)
              if lbl in syn_data and syn_data[lbl]["lat_ms"] is not None]

if lat_vals:
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    bars = ax3.bar(range(len(lat_labels)), lat_vals, color=lat_colors, alpha=0.88, width=0.55, edgecolor="white")
    for bar, v in zip(bars, lat_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f} ms", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax3.set_xticks(range(len(lat_labels)))
    ax3.set_xticklabels(lat_labels, fontsize=9, ha="center")
    ax3.set_ylabel("HLS Latency (ms) — one 64-step invocation", fontsize=10)
    ax3.set_title("Synthesis Latency per Configuration", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.set_ylim(0, max(lat_vals) * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "syn_latency.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: syn_latency.png")
else:
    print("No synthesis latency data found — skipping syn_latency.png")

# ---------------------------------------------------------------------------
# Plot 4: Resource Utilization Grouped Bar
# ---------------------------------------------------------------------------
RES_KEYS = ["LUT", "FF", "DSP", "BRAM_18K"]
RES_LABELS = ["LUT", "FF", "DSP", "BRAM"]
RES_COLORS = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]

util_configs = [(lbl, disp) for lbl, disp in SYN_CONFIGS
                if lbl in syn_data and syn_data[lbl]["util"]]
if util_configs:
    n_configs = len(util_configs)
    n_res     = len(RES_KEYS)
    x4        = np.arange(n_res)
    bar_w     = 0.8 / n_configs

    fig4, ax4 = plt.subplots(figsize=(11, 5.5))
    for i, (lbl, disp) in enumerate(util_configs):
        d = syn_data[lbl]
        pcts = []
        for rk in RES_KEYS:
            u = d["util"].get(rk, 0)
            a = d["avail"].get(rk, 1) or 1
            pcts.append(100.0 * u / a)
        offset = (i - n_configs/2 + 0.5) * bar_w
        bars = ax4.bar(x4 + offset, pcts, width=bar_w * 0.92,
                       label=disp.replace("\n", " "), color=COLORS6[i], alpha=0.85)

    ax4.set_xticks(x4)
    ax4.set_xticklabels(RES_LABELS, fontsize=11)
    ax4.set_ylabel("Utilization (%)", fontsize=10)
    ax4.set_title("Resource Utilization by Configuration (% of KV260 capacity)",
                  fontsize=11, fontweight="bold")
    ax4.legend(fontsize=8, loc="upper right", ncol=2)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")
    ax4.set_ylim(0, min(110, ax4.get_ylim()[1] * 1.15))
    ax4.axhline(100, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="100% cap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "syn_utilization.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: syn_utilization.png")
else:
    print("No utilization data found — skipping syn_utilization.png")

# ===========================================================================
# FPGA RUNTIME + ACCURACY PLOTS
# ===========================================================================

FPGA_DIR = os.path.join(SCRIPT_DIR, "fpga")

def parse_fpga_result(label):
    path = os.path.join(FPGA_DIR, f"fpga_{label}.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()

    passed = "PASS" in text and "FAIL" not in text.replace("*** Kria Host PASS", "")

    def _nums(pattern):
        m = re.search(pattern, text)
        if not m:
            return None, None, None
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    st_max, st_mean, st_rmse = _nums(
        r"state_out max / mean / rmse:\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)")
    o_max, o_mean, o_rmse = _nums(
        r"(?<!state_)out\s+max / mean / rmse:\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)")

    # FPGA-only total time (10 sequences)
    m_fpga = re.search(r"fpga-only\s*:\s*\d+ calls;\s*([\d.]+)\s*msecs", text)
    fpga_total_ms = float(m_fpga.group(1)) if m_fpga else None

    # End-to-end time
    m_e2e = re.search(r"host end-to-end\s*:\s*\d+ calls;\s*([\d.]+)\s*msecs", text)
    e2e_total_ms = float(m_e2e.group(1)) if m_e2e else None

    # Mean TPOT from per-seq lines
    tpots = re.findall(r"TPOT:\s*([\d.]+)\s*ms/step", text)
    mean_tpot = np.mean([float(t) for t in tpots]) if tpots else None

    return dict(
        passed=passed,
        out_max=o_max, out_mean=o_mean, out_rmse=o_rmse,
        st_max=st_max, st_mean=st_mean, st_rmse=st_rmse,
        fpga_total_ms=fpga_total_ms,
        e2e_total_ms=e2e_total_ms,
        mean_tpot=mean_tpot,
    )

FPGA_CONFIGS = [
    ("baseline",            "Baseline"),
    ("pragma",              "Pragma"),
    ("fixed_pwl",           "Fixed+PWL"),
    ("fixed_cordic",        "Fixed+CORDIC"),
    ("pragma_fixed_pwl",    "Pragma+\nFixed+PWL"),
    ("pragma_fixed_cordic", "Pragma+\nFixed+CORDIC"),
]

fpga_data = {}
for lbl, _ in FPGA_CONFIGS:
    d = parse_fpga_result(lbl)
    if d:
        fpga_data[lbl] = d

# ---------------------------------------------------------------------------
# Plot 5: FPGA Stacked Runtime per Step
# Bottom = HLS synthesis latency / 64 steps  (true FPGA compute)
# Top    = (measured fpga-only per step) - synthesis per step  (memory/AXI overhead)
# ---------------------------------------------------------------------------
N_STEPS = 64
N_SEQS  = 10

valid_stack = [(lbl, disp) for lbl, disp in FPGA_CONFIGS
               if lbl in fpga_data
               and fpga_data[lbl]["fpga_total_ms"] is not None
               and lbl in syn_data
               and syn_data[lbl].get("lat_ms") is not None]

if valid_stack:
    stack_labels  = [disp for _, disp in valid_stack]
    syn_per_step  = [syn_data[lbl]["lat_ms"] / N_STEPS for lbl, _ in valid_stack]
    meas_per_step = [fpga_data[lbl]["fpga_total_ms"] / (N_SEQS * N_STEPS) for lbl, _ in valid_stack]
    mem_overhead  = [max(0.0, m - s) for m, s in zip(meas_per_step, syn_per_step)]
    stack_colors_used = [COLORS6[i] for i, (lbl, _) in enumerate(FPGA_CONFIGS)
                         if lbl in dict(valid_stack)]

    x5 = np.arange(len(stack_labels))
    fig5, ax5 = plt.subplots(figsize=(11, 5.5))
    ax5.bar(x5, syn_per_step, color=stack_colors_used, alpha=0.9,
            label="FPGA Compute (synthesis)", edgecolor="white", width=0.55)
    ax5.bar(x5, mem_overhead, bottom=syn_per_step,
            color="lightgrey", alpha=0.75, label="Memory / AXI Transfer Overhead",
            edgecolor="white", hatch="//", width=0.55)

    # Annotate compute inside bar
    for i, v in enumerate(syn_per_step):
        ax5.text(i, v / 2, f"{v:.3f}\nms/step",
                 ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
    # Annotate total measured above bar
    for i, (s, m) in enumerate(zip(syn_per_step, mem_overhead)):
        ax5.text(i, s + m + 0.003, f"{s+m:.3f} meas.",
                 ha="center", va="bottom", fontsize=7, color="#555555")

    ax5.set_xticks(x5)
    ax5.set_xticklabels(stack_labels, fontsize=9)
    ax5.set_ylabel("Time per Step (ms/step)", fontsize=10)
    ax5.set_title("FPGA Runtime per Step: Compute vs Memory/AXI Overhead\n"
                  "Compute = HLS synthesis estimate  ·  Overhead = measured − synthesis",
                  fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9, loc="upper right")
    ax5.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fpga_runtime_stacked.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fpga_runtime_stacked.png")
else:
    print("No FPGA+synthesis data — skipping fpga_runtime_stacked.png")



# ---------------------------------------------------------------------------
# Plot 6: FPGA Accuracy Grouped Bar (max / mean / rmse)
# ---------------------------------------------------------------------------
acc_configs = [(lbl, disp) for lbl, disp in FPGA_CONFIGS
               if lbl in fpga_data and fpga_data[lbl]["out_mean"] is not None]
if acc_configs:
    acc_labels = [disp for _, disp in acc_configs]
    acc_max    = [fpga_data[lbl]["out_max"]  for lbl, _ in acc_configs]
    acc_mean   = [fpga_data[lbl]["out_mean"] for lbl, _ in acc_configs]
    acc_rmse   = [fpga_data[lbl]["out_rmse"] for lbl, _ in acc_configs]

    x6 = np.arange(len(acc_labels))
    w6 = 0.24
    fig6, ax6 = plt.subplots(figsize=(max(9, 2.5 * len(acc_labels)), 5.5))
    ax6.bar(x6 - w6, acc_max,  width=w6, label="Max error",  color="#e74c3c", alpha=0.88)
    ax6.bar(x6,      acc_mean, width=w6, label="Mean error", color="#3498db", alpha=0.88)
    ax6.bar(x6 + w6, acc_rmse, width=w6, label="RMSE",       color="#2ecc71", alpha=0.88)
    ax6.set_yscale("log")
    ax6.yaxis.set_major_formatter(LogFormatterMathtext())
    ax6.set_xticks(x6)
    ax6.set_xticklabels(acc_labels, fontsize=9, ha="center")
    ax6.set_ylabel("Output Error vs. Golden (log scale)", fontsize=10)
    ax6.set_title("FPGA Accuracy: Output Error vs. PyTorch Golden\n(all 6 configurations, 10 sequences × 64 steps)",
                  fontsize=11, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.35, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fpga_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: fpga_accuracy.png")
else:
    print("No FPGA accuracy data — skipping fpga_accuracy.png")

print(f"\nAll plots written to: {PLOT_DIR}/")

