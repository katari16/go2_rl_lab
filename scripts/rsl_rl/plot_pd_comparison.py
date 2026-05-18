"""Plot R1 (Kp=8, Kd=0.4) vs P20 (Kp=25, Kd=0.5) training curves from tensorboard logs.

Produces a 2x2 figure showing:
  - Policy/mean_noise_std    (exploration collapse)
  - Loss/entropy             (same story, independent)
  - Episode_Reward/joint_torques_l2  (peak torques — low PD wins by construction)
  - Episode_Reward/action_rate_l2    (control smoothness)

Saves PNG + PDF to scripts/rsl_rl/figures/pd_comparison.{png,pdf}.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS = {
    "R1 (Kp=8, Kd=0.4)": {
        "path": "/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_R1_h30_6d_big_tcn_norec/2026-04-21_19-15-56/",
        "color": "#1b9e77",
    },
    "P20 (Kp=25, Kd=0.5)": {
        "path": "/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P20_h30_4d_pd25/2026-04-19_11-10-59/",
        "color": "#d95f02",
    },
}

PANELS = [
    ("Train/mean_reward",                "Mean episode reward",           "Train/mean_reward"),
    ("Policy/mean_noise_std",            "Action noise std (σ)",          "Policy/mean_noise_std"),
    ("Episode_Reward/joint_torques_l2",  "L2 penalty (reward term)",      "Episode_Reward/joint_torques_l2"),
    ("Episode_Reward/action_rate_l2",    "L2 penalty (reward term)",      "Episode_Reward/action_rate_l2"),
]


def load_scalar(path: str, tag: str):
    ea = EventAccumulator(path, size_guidance={"scalars": 100000})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.float64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values


def smooth(y: np.ndarray, window: int = 21) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.flatten()

    for ax, (tag, ylabel, title) in zip(axes, PANELS):
        for label, info in RUNS.items():
            steps, vals = load_scalar(info["path"], tag)
            if steps is None:
                ax.text(0.5, 0.5, f"missing: {tag}", transform=ax.transAxes, ha="center")
                continue
            ax.plot(steps, vals, color=info["color"], alpha=0.25, linewidth=0.8)
            ax.plot(steps, smooth(vals), color=info["color"], linewidth=1.8, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.25)

    for ax in axes[2:]:
        ax.set_xlabel("Training iteration", fontsize=10)

    # Single legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=11)

    fig.suptitle("PD gain comparison: low (Kp=8, Kd=0.4) vs default (Kp=25, Kd=0.5)",
                 fontsize=12, y=1.06)
    fig.tight_layout()

    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)
    png_path = out_dir / "pd_comparison.png"
    pdf_path = out_dir / "pd_comparison.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
