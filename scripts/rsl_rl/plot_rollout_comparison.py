"""Plot rollout estimator comparison across ablation runs.

Reads data/eval/master_metrics.json (produced by collect_rollout_metrics.py)
and generates a PDF with horizontal bar charts for each metric.

Panels:
  Row 1: Fx MAE | Fy MAE | Fz MAE | τ_yaw MAE
  Row 2: Force MAE total | Range-norm Force MAE % | Median Angular Error | Relative Err %

Bars are colored by series (A=blue, B=green, J=orange, P=gradient).
NaN bars (axis not estimated by that model) are shown as empty.

Usage:
    python scripts/rsl_rl/plot_rollout_comparison.py
    python scripts/rsl_rl/plot_rollout_comparison.py --input data/eval/master_metrics.json --out data/eval/comparison.pdf
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Color scheme ─────────────────────────────────────────────────────────────

def _run_color(run_id: str) -> str:
    if run_id.startswith("A"):
        return "#4472C4"
    if run_id.startswith("B"):
        return "#70AD47"
    if run_id.startswith("J"):
        return "#FF7043"
    # P-series: gradient from light to dark using index
    p_num = int(run_id[1:])
    # map P1..P20 (excl P18) to [0,1]
    p_all = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20]
    idx = p_all.index(p_num) if p_num in p_all else 0
    t = idx / max(len(p_all) - 1, 1)
    cmap = plt.cm.Blues
    r, g, b, _ = cmap(0.35 + 0.55 * t)
    return (r, g, b)


# ── Metric extraction ─────────────────────────────────────────────────────────

def extract(run: dict) -> dict:
    """Pull flat metrics from a master_metrics entry."""
    basket = run["metrics"].get("baskets", {}).get("training", {})
    tr     = run["metrics"].get("training_regime", {})
    pam    = basket.get("per_axis_mae", {})
    max_f  = tr.get("max_force", 30.0)
    force_mae = basket.get("force_mae")

    return {
        "Fx":           pam.get("Fx"),
        "Fy":           pam.get("Fy"),
        "Fz":           pam.get("Fz"),
        "τ_yaw":        pam.get("τ_yaw"),
        "τ_roll":       pam.get("τ_roll"),
        "τ_pitch":      pam.get("τ_pitch"),
        "force_mae":    force_mae,
        "torque_mae":   basket.get("torque_mae"),
        "range_norm_pct": (force_mae / max_f * 100) if force_mae is not None else None,
        "ang_err_median": basket.get("angular_err_xy_deg_median"),
        "rel_err_pct":  basket.get("relative_err_no_mask_pct"),
        "max_force":    max_f,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _hbar(ax, run_ids, values, colors, xlabel, title, vline=None, fmt=".2f"):
    """Draw a horizontal bar chart. NaN/None values shown as empty gray bars."""
    n = len(run_ids)
    y = np.arange(n)
    bar_h = 0.7

    valid = [v for v in values if v is not None and not math.isnan(v)]
    xmax = max(valid) * 1.22 if valid else 1.0
    label_offset = xmax * 0.012

    for i, (v, c) in enumerate(zip(values, colors)):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            ax.barh(i, 0, bar_h, color="lightgray", edgecolor="none")
        else:
            ax.barh(i, v, bar_h, color=c, edgecolor="white", linewidth=0.4)
            ax.text(v + label_offset, i, f"{v:{fmt}}",
                    va="center", ha="left", fontsize=7.5)

    if vline is not None:
        ax.axvline(vline, color="black", lw=0.8, ls="--", alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(run_ids, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlim(0, xmax)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Run description table ────────────────────────────────────────────────────

# Static per-run metadata for the description page.
# Columns: (group, est_dims, history, max_force, max_torque, network, special)
RUN_DESC = {
    # A-series — 3D XYZ force estimation, older batch-1 runs (20 N)
    "A1": ("A · 3D history sweep",  "3D (Fx,Fy,Fz)",          "h=10",  "20 N", "—",      "enc[128,64]",   "rec loss w=1"),
    "A2": ("A · 3D history sweep",  "3D (Fx,Fy,Fz)",          "h=40",  "20 N", "—",      "enc[128,64]",   "rec loss w=1"),
    # B-series — 6D full wrench estimation (20 N + 5 Nm)
    "B2": ("B · 6D wrench",         "6D (F+τ roll/pitch/yaw)", "h=40",  "20 N", "5 Nm",   "enc[128,64]",   "rec loss w=1"),
    "B3": ("B · 6D wrench (big)",   "6D (F+τ roll/pitch/yaw)", "h=40",  "20 N", "5 Nm",   "enc[256,128]",  "rec loss w=1, bigger net"),
    # J-series — 4D with estimation-accuracy reward (30 N + 10 Nm)
    "J1": ("J · est-reward sweep",  "4D (Fx,Fy,Fz,τ_yaw)",    "h=40",  "30 N", "10 Nm",  "enc[128,64]",   "est-accuracy reward w=1"),
    "J3": ("J · est-reward sweep",  "4D (Fx,Fy,Fz,τ_yaw)",    "h=40",  "30 N", "10 Nm",  "enc[128,64]",   "est-accuracy reward w=50"),
    "J5": ("J · TCN preprocessor",  "4D (Fx,Fy,Fz,τ_yaw)",    "h=40",  "30 N", "10 Nm",  "TCN pre [64,64]","est-accuracy reward w=50, TCN preprocessor"),
    # P-series — PAINT trapezoid wrench (30 N + 10 Nm unless noted)
    "P1":  ("P · history sweep",    "4D (Fx,Fy,Fz,τ_yaw)",    "h=10",  "30 N", "10 Nm",  "enc[128,64]",   "PAINT trapezoid wrench"),
    "P2":  ("P · history sweep",    "4D (Fx,Fy,Fz,τ_yaw)",    "h=20",  "30 N", "10 Nm",  "enc[128,64]",   "PAINT trapezoid wrench"),
    "P3":  ("P · history sweep",    "4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "PAINT wrench [BASELINE]"),
    "P4":  ("P · history sweep",    "4D (Fx,Fy,Fz,τ_yaw)",    "h=40",  "30 N", "10 Nm",  "enc[128,64]",   "PAINT trapezoid wrench"),
    "P5":  ("P · network size",     "4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[64,32]",    "smaller network (half)"),
    "P6":  ("P · network size",     "4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[256,128]",  "bigger network (double)"),
    "P7":  ("P · est-acc reward",   "4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "est-accuracy reward w=50"),
    "P8":  ("P · compliance reward","4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "compliance reward w=0.5"),
    "P9":  ("P · compliance reward","4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "compliance reward w=1.0"),
    "P10": ("P · compliance reward","4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "compliance reward w=5.0"),
    "P11": ("P · architecture",     "4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "no reconstruction loss"),
    "P12": ("P · architecture",     "4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "TCN enc[64,128]","TCN encoder (not preprocessor)"),
    "P13": ("P · force dim",        "2D (Fx,Fy)",              "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "2D estimator only"),
    "P14": ("P · force dim",        "3D xy+yaw (Fx,Fy,τ_yaw)","h=30",  "30 N", "10 Nm",  "enc[128,64]",   "xy_yaw layout, yaw loss w=3"),
    "P15": ("P · force dim",        "4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "dim-ablation control (≡ P3)"),
    "P16": ("P · force dim",        "6D (F+τ roll/pitch/yaw)", "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "torque angle loss w=3, yaw loss w=3"),
    "P17": ("P · force dim",        "6D (F+τ roll/pitch/yaw)", "h=30",  "30 N", "10 Nm",  "enc[256,128]",  "6D bigger net, torque angle loss w=3"),
    "P19": ("P · torque range",     "4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "5 Nm",   "enc[128,64]",   "lower torque range (5 Nm vs 10 Nm)"),
    "P20": ("P · PD gains",         "4D (Fx,Fy,Fz,τ_yaw)",    "h=30",  "30 N", "10 Nm",  "enc[128,64]",   "stiffer PD: Kp=25, Kd=0.5"),
}

_COL_HEADERS = ["Run", "Group / Ablation", "Est. dimensions", "History",
                "Max F", "Max τ", "Network", "Special / notes"]
_COL_WIDTHS  = [0.04,   0.17,               0.19,              0.06,
                0.06,   0.06,    0.11,       0.31]   # fraction of fig width


def _make_description_page(master: dict, run_ids: list, colors: list) -> plt.Figure:
    """Build a full-page table describing every run."""
    fig = plt.figure(figsize=(22, len(run_ids) * 0.45 + 2.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.text(0.5, 0.97, "Ablation Run Descriptions",
             ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.945,
             "Training regime: PAINT trapezoid wrench unless noted. "
             "All runs: no active mask, 128 envs, 10 s rollout.",
             ha="center", va="top", fontsize=9, color="#444")

    n = len(run_ids)
    row_h   = 0.82 / (n + 1)   # fractional height per row (header + n data rows)
    y_start = 0.92              # top of table in figure coordinates
    x_cols  = []                # left edge of each column
    x = 0.01
    for w in _COL_WIDTHS:
        x_cols.append(x)
        x += w

    # ── Header row ────────────────────────────────────────────────────────────
    y_hdr = y_start - row_h * 0.15
    for xi, hdr in zip(x_cols, _COL_HEADERS):
        fig.text(xi + 0.005, y_hdr, hdr, va="center", ha="left",
                 fontsize=9, fontweight="bold", color="white",
                 bbox=dict(boxstyle="square,pad=0.3", fc="#2c3e50", ec="none"))

    # ── Data rows ─────────────────────────────────────────────────────────────
    for row_i, (rid, color) in enumerate(zip(run_ids, colors)):
        y = y_start - row_h * (row_i + 1.1)
        bg = "#f0f4f8" if row_i % 2 == 0 else "white"

        # Row background
        fig.patches.append(plt.Rectangle(
            (0.005, y - row_h * 0.42), 0.99, row_h * 0.85,
            transform=fig.transFigure, fc=bg, ec="none", zorder=0,
        ))

        desc = RUN_DESC.get(rid, {})
        if isinstance(desc, tuple):
            group, est_dims, hist, max_f, max_t, net, special = desc
        else:
            group = est_dims = hist = max_f = max_t = net = special = "—"

        cells = [rid, group, est_dims, hist, max_f, max_t, net, special]
        for col_i, (xi, cell) in enumerate(zip(x_cols, cells)):
            is_run_id = (col_i == 0)
            fw = "bold" if is_run_id else "normal"
            fc = "white" if is_run_id else "#222"
            bbox_kw = (dict(boxstyle="round,pad=0.25", fc=color, ec="none")
                       if is_run_id else None)
            fig.text(xi + 0.005, y, str(cell),
                     va="center", ha="left", fontsize=8,
                     fontweight=fw, color=fc,
                     bbox=bbox_kw)

        # Thin separator line
        fig.add_artist(plt.Line2D(
            [0.005, 0.995], [y - row_h * 0.43, y - row_h * 0.43],
            transform=fig.transFigure, color="#ccc", lw=0.4, zorder=1,
        ))

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/eval/master_metrics.json")
    parser.add_argument("--out",   default="data/eval/comparison.pdf")
    args = parser.parse_args()

    with open(args.input) as f:
        master = json.load(f)

    # Preserve insertion order (same as RUNS list in collect script)
    run_ids = list(master.keys())
    runs    = [master[r] for r in run_ids]
    metrics = [extract(r) for r in runs]
    colors  = [_run_color(r) for r in run_ids]

    def vals(key):
        return [m[key] for m in metrics]

    # ── Figure layout: 2 rows × 4 cols ───────────────────────────────────────
    n_runs = len(run_ids)
    bar_h_in = 0.32   # inches per bar
    fig_h = max(10, n_runs * bar_h_in + 5)
    fig, axes = plt.subplots(2, 4, figsize=(24, fig_h))
    fig.suptitle("Rollout Estimator Comparison — Training Regime",
                 fontsize=14, fontweight="bold", y=1.01)

    # Row 1 — per-axis MAE
    _hbar(axes[0, 0], run_ids, vals("Fx"),    colors, "MAE (N)",  "Fx MAE")
    _hbar(axes[0, 1], run_ids, vals("Fy"),    colors, "MAE (N)",  "Fy MAE")
    _hbar(axes[0, 2], run_ids, vals("Fz"),    colors, "MAE (N)",  "Fz MAE\n(N/A if not estimated)")
    _hbar(axes[0, 3], run_ids, vals("τ_yaw"), colors, "MAE (Nm)", "τ_yaw MAE\n(N/A if not estimated)")

    # Row 2 — aggregate metrics
    _hbar(axes[1, 0], run_ids, vals("force_mae"),     colors, "MAE (N)",  "Force MAE\n(avg over force axes)")
    _hbar(axes[1, 1], run_ids, vals("range_norm_pct"),colors, "% of max_force", "Range-Norm Force MAE\n(force_mae / max_force × 100)")
    _hbar(axes[1, 2], run_ids, vals("ang_err_median"),colors, "degrees",  "Median Angular Error\n(XY plane, GT > 1N)")
    _hbar(axes[1, 3], run_ids, vals("rel_err_pct"),   colors, "%",        "Relative Error\n(|err| / |GT|, no 1N mask)")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color="#4472C4", label="A-series: 3D XYZ, 20N"),
        mpatches.Patch(color="#70AD47", label="B-series: 6D wrench, 20N+5Nm"),
        mpatches.Patch(color="#FF7043", label="J-series: 4D, est-accuracy reward, 30N+10Nm"),
        mpatches.Patch(color=_run_color("P1"), label="P-series (light): h=10–40 sweep"),
        mpatches.Patch(color=_run_color("P20"), label="P-series (dark): later ablations"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.0), fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

        desc_fig = _make_description_page(master, run_ids, colors)
        pdf.savefig(desc_fig, dpi=150, bbox_inches="tight")
        plt.close(desc_fig)

    print(f"Saved: {out_path}")

    # ── Quick terminal summary ────────────────────────────────────────────────
    print(f"\n{'Run':>4}  {'ForcMAE':>8}  {'RangeN%':>8}  {'AngMed°':>8}  {'RelErr%':>8}  axes")
    print("─" * 60)
    for rid, m in zip(run_ids, metrics):
        axes_str = "+".join(k for k in ("Fx","Fy","Fz","τ_yaw","τ_roll","τ_pitch")
                            if m.get(k) is not None)
        fm   = m["force_mae"];     fm_s   = f"{fm:.3f}"   if fm   is not None else "  n/a "
        rn   = m["range_norm_pct"];rn_s   = f"{rn:.1f}"   if rn   is not None else " n/a"
        ang  = m["ang_err_median"];ang_s  = f"{ang:.1f}"  if ang  is not None else " n/a"
        rel  = m["rel_err_pct"];   rel_s  = f"{rel:.1f}"  if rel  is not None else " n/a"
        print(f"{rid:>4}  {fm_s:>8}  {rn_s:>8}  {ang_s:>8}  {rel_s:>8}  {axes_str}")


if __name__ == "__main__":
    main()
