"""Horizontal bar chart of estimator metrics — 3 build-up PNGs for PowerPoint.

Generates 3 images with identical axes, each revealing one more group:
  step1 — MAE Fx, MAE Fy
  step2 — + MAE τ_yaw
  step3 — + Ang err mean, Ang err median

Usage:
    python scripts/plot_estimator_metrics.py <metrics.json> [--basket <key>] [--out_dir <dir>]
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("metrics", type=str)
parser.add_argument("--basket", type=str, default=None)
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()

mpl.rcParams["font.family"] = "Liberation Sans"
mpl.rcParams["font.size"] = 14

with open(args.metrics) as f:
    data = json.load(f)

baskets = data.get("baskets", data)
key = next((k for k in baskets if str(args.basket) in k), list(baskets.keys())[0]) if args.basket else list(baskets.keys())[0]
m = baskets[key]
pam  = m["per_axis_mae"]
pams = m.get("per_axis_mae_std", {})

# ── All 5 entries (no Fz) ──────────────────────────────────────────────────
labels = [
    "MAE $F_x$",
    "MAE $F_y$",
    r"MAE $\tau_{yaw}$",
    "Ang. err mean (XY)",
    "Ang. err median (XY)",
]
values = [
    pam["Fx"],
    pam["Fy"],
    pam["τ_yaw"],
    m.get("angular_err_xy_deg_mean", 0.0),
    m.get("angular_err_xy_deg_median", 0.0),
]
errors = [
    pams.get("Fx", 0.0),
    pams.get("Fy", 0.0),
    pams.get("τ_yaw", 0.0),
    m.get("angular_err_xy_deg_mean_std", 0.0),
    m.get("angular_err_xy_deg_median_std", 0.0),
]
units  = ["N", "N", "Nm", "deg", "deg"]
colors = ["#1d3557", "#457b9d", "#2a6496", "#a8dadc", "#74c2e1"]

# Groups revealed per step
GROUPS = [
    {0, 1},        # step 1: Fx, Fy
    {0, 1, 2},     # step 2: + τ_yaw
    {0, 1, 2, 3, 4},  # step 3: + ang mean + ang median
]

x_max = 13.0
y = np.arange(len(labels))

out_dir = args.out_dir or os.path.dirname(args.metrics)

for step_idx, visible in enumerate(GROUPS):
    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(len(labels)):
        if i in visible:
            ax.barh(y[i], values[i], xerr=errors[i], color=colors[i], alpha=0.88,
                    error_kw=dict(ecolor="black", elinewidth=1.4, capsize=5, capthick=1.4),
                    height=0.55)
            x_pos = values[i] + errors[i] + x_max * 0.02
            ax.text(x_pos, i, f"{values[i]:.2f}", va="center", fontsize=12,
                    color="#222222", fontweight="bold")
            ax.text(x_pos + x_max * 0.10, i, f"± {errors[i]:.2f} {units[i]}",
                    va="center", fontsize=12, color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("Error", fontsize=14)
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout(pad=1.2)
    out = os.path.join(out_dir, f"metrics_bar_step{step_idx + 1}.png")
    plt.savefig(out, dpi=150)  # no bbox_inches="tight" — keeps identical canvas
    plt.close()
    print(f"Saved: {out}")
