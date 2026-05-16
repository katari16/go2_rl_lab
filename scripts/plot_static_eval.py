"""Plot static eval data from saved JSON.

Usage:
    python scripts/plot_static_eval.py <path_to_json> [--show_est] [--out <path>]
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="Path to static_eval_data_*.json")
parser.add_argument("--show_est", action="store_true", default=False,
                    help="Overlay estimated force on top of GT.")
parser.add_argument("--out", type=str, default=None,
                    help="Output path for saved figure (default: next to JSON).")
args = parser.parse_args()

with open(args.data) as f:
    d = json.load(f)

t           = np.array(d["time_s"])
rerandom    = d["rerandom_steps"]

# Trim to first 15 seconds
mask = t <= 15.0
t = t[mask]

channels = [
    ("$F_x$",       "gt_force_x",   "est_force_x",   "N",  "#e63946"),
    ("$F_y$",       "gt_force_y",   "est_force_y",   "N",  "#9b2226"),
    ("$F_z$",       "gt_force_z",   "est_force_z",   "N",  "#5c4db1"),
    (r"$\tau_{yaw}$", "gt_torque_yaw", "est_torque_yaw", "Nm", "#6d3a1e"),
]

# filter to channels that exist in the data
channels = [c for c in channels if c[1] in d]

fig, axes = plt.subplots(len(channels), 1, figsize=(13, 3 * len(channels)), sharex=True)
if len(channels) == 1:
    axes = [axes]

title = "Ground Truth Force Profile" if not args.show_est else "GT vs Estimated Force"
fig.suptitle(title, fontsize=13, fontweight="bold")

for ax, (label, gt_key, est_key, unit, color) in zip(axes, channels):
    gt = np.array(d[gt_key])[mask]
    ax.step(t, gt, where="post", color=color, linewidth=2.0, label=f"GT {label}")

    if args.show_est and est_key in d:
        est = np.array(d[est_key])[mask]
        ax.plot(t, est, color=color, linewidth=1.0, alpha=0.6,
                linestyle="--", label=f"Est {label}")
        ax.legend(loc="upper right", fontsize=8)

    for rs in rerandom:
        idx = int(rs)
        if idx < len(t):
            ax.axvline(t[idx], color="gray", linewidth=0.5,
                       linestyle="--", alpha=0.4)

    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)
    ax.set_ylabel(f"{label} ({unit})", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Time (s)", fontsize=11)
plt.tight_layout()

if args.out:
    out = args.out
else:
    base = os.path.splitext(args.data)[0]
    suffix = "_gt_est" if args.show_est else "_gt"
    out = base + suffix + ".png"

plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
