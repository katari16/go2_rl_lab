"""Plot P1 vs J3 estimator comparison on shared GT axes.

Colours (add legend in PowerPoint):
  GT      — per-channel solid colour (red Fx, dark red Fy, brown τ_yaw)
  P1 est  — #9b5de5  (purple)
  J3 est  — #f4a261  (orange)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

P1_PATH = (
    "/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series"
    "/ablation_P1_h10_4d/2026-04-19_11-13-01"
    "/force_eval/static_eval_data_2026-05-03_16-06-16.json"
)
J3_PATH = (
    "/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_force_accuracy_reward"
    "/ablation_J3_4d_h40_estrew_w50_30N/2026-04-15_17-24-50"
    "/force_eval/static_eval_data_2026-05-03_15-12-19.json"
)

with open(P1_PATH) as f:
    d_p1 = json.load(f)
with open(J3_PATH) as f:
    d_j3 = json.load(f)

t    = np.array(d_p1["time_s"])
mask = t <= 15.0
t    = t[mask]
rerandom = d_p1["rerandom_steps"]

COLOR_P1 = "#9b5de5"
COLOR_J3 = "#f4a261"

channels = [
    ("$F_x$",         "gt_force_x",    "est_force_x",    "N",  "#e63946"),
    ("$F_y$",         "gt_force_y",    "est_force_y",    "N",  "#9b2226"),
    (r"$\tau_{yaw}$", "gt_torque_yaw", "est_torque_yaw", "Nm", "#6d3a1e"),
]

fig, axes = plt.subplots(len(channels), 1, figsize=(13, 3 * len(channels)), sharex=True)

for ax, (label, gt_key, est_key, unit, gt_color) in zip(axes, channels):
    gt = np.array(d_p1[gt_key])[mask]
    ax.plot(t, np.array(d_p1[est_key])[mask], color=COLOR_P1, linewidth=1.8, alpha=0.9)
    ax.plot(t, np.array(d_j3[est_key])[mask], color=COLOR_J3, linewidth=1.8, alpha=0.9)
    ax.step(t, gt, where="post", color=gt_color, linewidth=2.2, zorder=5)

    for rs in rerandom:
        idx = int(rs)
        if idx < len(t):
            ax.axvline(t[idx], color="gray", linewidth=0.5, linestyle="--", alpha=0.4)

    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--", alpha=0.4)
    ax.set_ylabel(f"{label} ({unit})", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Time (s)", fontsize=11)
plt.tight_layout()

out = os.path.join(os.path.dirname(J3_PATH), "..", "p1_vs_j3_comparison.png")
out = os.path.normpath(out)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
