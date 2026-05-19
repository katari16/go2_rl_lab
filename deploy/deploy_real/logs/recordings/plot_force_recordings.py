"""Plot force magnitude timeseries from real-robot recordings.

Takes the latest recording CSV per ablation, plots estimated force_mag vs GT,
and shows MAE in the legend. Default GT = 2.09 kg x 9.81 m/s^2 = 20.50 N
(pulley payload).

Usage:
    python plot_force_recordings.py
    python plot_force_recordings.py --gt 20.50 --out force_mag_comparison.png
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

RECORDINGS_DIR = Path(__file__).resolve().parent

# Pretty labels per config stem
LABELS = {
    # Group 1 — history / net / TCN / PD gains (4D force)
    "go2_ablation_p1":  "P1 (4D, h=10, baseline)",
    "go2_ablation_p2":  "P2 (4D, h=20, baseline)",
    "go2_ablation_p3":  "P3 (4D, h=30, baseline)",
    "go2_ablation_p4":  "P4 (4D, h=40, baseline)",
    "go2_ablation_p5":  "P5 (4D, h=30, enc=[64,32])",
    "go2_ablation_p6":  "P6 (4D, h=30, enc=[256,128])",
    "go2_ablation_p12": "P12 (4D, h=30, TCN encoder)",
    "go2_ablation_p20": "P20 (4D, h=30, Kp=25 Kd=0.5)",
    # Earlier J-series
    "go2_ablation_j3":  "J3 (4D, h=40, est_acc w=50)",
    "go2_ablation_j5":  "J5 (4D, h=40, TCN + est_acc w=50)",
    # Rewards
    "go2_ablation_p9":  "P9 (4D, h=30, compliance w=1)",
    "go2_ablation_p10": "P10 (4D, h=30, compliance w=5)",
    # 6D / payload
    "go2_ablation_p17": "P17 (6D, h=30, enc=[256,128])",
    "go2_ablation_p18": "P18 (6D, payload 0-4kg)",
    "go2_payload_3kg":  "Payload 3kg (3D, h=20)",
    # 6Dctrl-Total50 family
    "go2_ablation_6dctrl_total50":            "6Dctrl-Total50 (no est-acc)",
    "go2_ablation_6dctrl_total50_estacc_w10": "6Dctrl-Total50 (est-acc w=10)",
    "go2_ablation_6dctrl_total50_estacc_w25": "6Dctrl-Total50 (est-acc w=25)",
    "go2_ablation_6dctrl_total50_estacc_w50": "6Dctrl-Total50 (est-acc w=50)",
}


def find_latest_per_run(recordings_dir: Path, n_per_run: int = 1) -> dict:
    """Return {stem: [csv_path, ...]} with the last n_per_run recordings per stem,
    ordered oldest -> newest."""
    by_stem = defaultdict(list)
    for f in recordings_dir.glob("*.csv"):
        m = re.match(r"(go2_\w+?)_rec\d+_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv", f.name)
        if m:
            by_stem[m.group(1)].append((m.group(2), f))
    out = {}
    for stem, files in by_stem.items():
        sorted_files = [p for _, p in sorted(files)]
        out[stem] = sorted_files[-n_per_run:]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=float, default=20.50,
                        help="Ground truth force magnitude (N). Default = 2.09 kg x 9.81 m/s^2 = 20.50 N.")
    parser.add_argument("--out", type=str, default="force_mag_comparison.png", help="Output filename")
    parser.add_argument("--n_per_run", type=int, default=2,
                        help="Plot the last N recordings per ablation (default 2).")
    args = parser.parse_args()

    latest = find_latest_per_run(RECORDINGS_DIR, n_per_run=args.n_per_run)
    if not latest:
        print("No recording CSVs found.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.tab10.colors
    linestyles = ["--", "-", ":", "-."]  # oldest -> newest
    for idx, (stem, csv_paths) in enumerate(sorted(latest.items())):
        color = colors[idx % len(colors)]
        label_base = LABELS.get(stem, stem)
        k = len(csv_paths)
        for j, csv_path in enumerate(csv_paths):
            df = pd.read_csv(csv_path)
            t = df["t"].values
            mag = df["force_mag"].values
            mask = t >= 5.0
            t, mag = t[mask], mag[mask]
            if t.size == 0:
                continue
            t = t - t[0]

            mae = np.mean(np.abs(mag - args.gt))
            # Oldest recording gets the earliest style, newest gets solid.
            ls = linestyles[-(k - j)] if k <= len(linestyles) else "-"
            tag = f"run {j + 1}/{k}"
            label = f"{label_base} ({tag})  |  MAE={mae:.2f}N"
            ax.plot(t, mag, color=color, linestyle=ls, linewidth=1.2, alpha=0.85, label=label)

    ax.axhline(args.gt, color="black", linewidth=1.8, linestyle="--", label=f"GT = {args.gt:.2f}N")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Estimated Force Magnitude (N)", fontsize=12)
    ax.set_title(f"Real-Robot Force Magnitude Estimation — GT={args.gt:.2f}N (2.09 kg payload)", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = RECORDINGS_DIR / args.out
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Print MAE summary table — one row per recording
    print(f"\n{'Run':<50} {'Steps':>6}  {'MAE (N)':>8}  {'Mean (N)':>9}  {'Std (N)':>8}")
    print("-" * 90)
    for stem, csv_paths in sorted(latest.items()):
        k = len(csv_paths)
        for j, csv_path in enumerate(csv_paths):
            df = pd.read_csv(csv_path)
            mag = df["force_mag"].values[df["t"].values >= 5.0]
            if mag.size == 0:
                continue
            mae = np.mean(np.abs(mag - args.gt))
            label = f"{LABELS.get(stem, stem)} (run {j + 1}/{k})"
            print(f"{label:<50} {len(mag):>6}  {mae:>8.2f}  {mag.mean():>9.2f}  {mag.std():>8.2f}")


if __name__ == "__main__":
    main()
