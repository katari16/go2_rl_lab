"""Analyze real-world deployment recordings.

Parses the recorded JSON data, splits it into recording segments based on
the sim_real_recording toggle, labels each segment with its compliance_mode
(normal / off / inverted), and generates per-segment time-series plots.

Usage:
    python scripts/analysis/analyze_real_deploy.py data/real_world_data/real_deploy_debug_2026-04-07_21-29-45.json
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

MODE_COLORS = {"normal": "#2196F3", "off": "#9E9E9E", "inverted": "#F44336"}
MODE_LABELS = {"normal": "Normal mapping", "off": "No mapping", "inverted": "Inverted mapping"}


def find_segments(recording_flags: list[bool]) -> list[tuple[int, int]]:
    """Find contiguous True segments in the recording flag."""
    segments = []
    in_seg = False
    for i, r in enumerate(recording_flags):
        if r and not in_seg:
            seg_start = i
            in_seg = True
        elif not r and in_seg:
            segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(recording_flags) - 1))
    return segments


def extract_segment(data: dict, start: int, end: int) -> dict:
    """Extract a segment slice from the full data dict."""
    sl = slice(start, end + 1)
    return {
        "time_s": np.array(data["time_s"][sl]),
        "mode": data["compliance_mode"][start],  # uniform within segment
        "force_hat": np.array(data["force_hat"][sl]),
        "force_ema": np.array(data["force_ema"][sl]),
        "velocity_cmd": np.array(data["velocity_cmd"][sl]),
        "raw_obs": np.array(data["raw_obs"][sl]),
        "actions": np.array(data["actions"][sl]),
        "start_idx": start,
        "end_idx": end,
    }


def plot_segment(seg: dict, seg_idx: int, obs_labels: list[str], pdf: PdfPages):
    """Generate plots for a single recording segment."""
    t = seg["time_s"] - seg["time_s"][0]  # relative time
    mode = seg["mode"]
    color = MODE_COLORS.get(mode, "#000000")
    label = MODE_LABELS.get(mode, mode)
    duration = t[-1]

    # ── Page 1: Force estimate + EMA + velocity commands ─────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Segment {seg_idx} — {label} — {duration:.1f}s "
        f"(steps {seg['start_idx']}–{seg['end_idx']})",
        fontsize=14, fontweight="bold",
    )

    # Force estimate
    ax = axes[0]
    ax.plot(t, seg["force_hat"][:, 0], label="F̂x", color="#E53935", alpha=0.8)
    ax.plot(t, seg["force_hat"][:, 1], label="F̂y", color="#1E88E5", alpha=0.8)
    ax.plot(t, seg["force_hat"][:, 2], label="F̂z", color="#43A047", alpha=0.8)
    ax.set_ylabel("Force estimate (N)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    # Force EMA
    ax = axes[1]
    ax.plot(t, seg["force_ema"][:, 0], label="EMA x", color="#E53935", alpha=0.8)
    ax.plot(t, seg["force_ema"][:, 1], label="EMA y", color="#1E88E5", alpha=0.8)
    if seg["force_ema"].shape[1] > 2:
        ax.plot(t, seg["force_ema"][:, 2], label="EMA z", color="#43A047", alpha=0.8)
    ax.set_ylabel("Force EMA (N)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    # Force magnitude
    force_mag = np.linalg.norm(seg["force_hat"][:, :2], axis=1)
    ax2 = ax.twinx()
    ax2.plot(t, force_mag, label="|F̂xy|", color="#FF9800", alpha=0.5, linewidth=1)
    ax2.set_ylabel("|F̂xy| (N)", color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    # Velocity commands
    ax = axes[2]
    ax.plot(t, seg["velocity_cmd"][:, 0], label="vx cmd", color="#E53935", alpha=0.8)
    ax.plot(t, seg["velocity_cmd"][:, 1], label="vy cmd", color="#1E88E5", alpha=0.8)
    ax.plot(t, seg["velocity_cmd"][:, 2], label="ωz cmd", color="#43A047", alpha=0.8)
    ax.set_ylabel("Velocity cmd")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 2: Angular velocity + gravity + torques ─────────────────────
    obs = seg["raw_obs"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Segment {seg_idx} — {label} — Proprioception", fontsize=14, fontweight="bold")

    # Angular velocity
    ax = axes[0]
    for i, name in enumerate(["ωx", "ωy", "ωz"]):
        ax.plot(t, obs[:, i], label=name, alpha=0.8)
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Gravity projection
    ax = axes[1]
    for i, name in enumerate(["gx", "gy", "gz"]):
        ax.plot(t, obs[:, 3 + i], label=name, alpha=0.8)
    ax.set_ylabel("Projected gravity")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Torques (obs indices 45:57, scaled by 0.1 in obs)
    ax = axes[2]
    torques = obs[:, 45:57] * 10.0  # undo 0.1 scaling
    torque_mag = np.linalg.norm(torques, axis=1)
    ax.plot(t, torque_mag, label="|τ|", color="#7B1FA2", alpha=0.8)
    ax.set_ylabel("Torque magnitude (Nm)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_overview(segments: list[dict], pdf: PdfPages):
    """Plot a full-session overview showing all segments in context."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Full Session Overview — Recording Segments", fontsize=14, fontweight="bold")

    for seg in segments:
        t = seg["time_s"]
        mode = seg["mode"]
        color = MODE_COLORS.get(mode, "#000000")
        label = MODE_LABELS.get(mode, mode)

        # Force magnitude
        force_mag = np.linalg.norm(seg["force_hat"][:, :2], axis=1)
        axes[0].plot(t, force_mag, color=color, alpha=0.8, linewidth=1)
        axes[0].axvspan(t[0], t[-1], alpha=0.1, color=color, label=label)

        # Velocity commands magnitude
        vel_mag = np.linalg.norm(seg["velocity_cmd"][:, :2], axis=1)
        axes[1].plot(t, vel_mag, color=color, alpha=0.8, linewidth=1)
        axes[1].axvspan(t[0], t[-1], alpha=0.1, color=color)

    axes[0].set_ylabel("|F̂xy| (N)")
    axes[0].grid(True, alpha=0.3)
    # Deduplicate legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc="upper right")

    axes[1].set_ylabel("|vel cmd xy|")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_summary_table(segments: list[dict], pdf: PdfPages):
    """Summary table of all segments."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    headers = ["Seg", "Mode", "Duration (s)", "Steps", "Mean |F̂xy| (N)",
               "Max |F̂xy| (N)", "Mean |vel cmd|"]
    rows = []
    for i, seg in enumerate(segments):
        t = seg["time_s"]
        force_mag = np.linalg.norm(seg["force_hat"][:, :2], axis=1)
        vel_mag = np.linalg.norm(seg["velocity_cmd"][:, :2], axis=1)
        rows.append([
            str(i),
            MODE_LABELS.get(seg["mode"], seg["mode"]),
            f"{t[-1] - t[0]:.1f}",
            str(seg["end_idx"] - seg["start_idx"] + 1),
            f"{force_mag.mean():.2f}",
            f"{force_mag.max():.2f}",
            f"{vel_mag.mean():.3f}",
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)

    # Color rows by mode
    for i, seg in enumerate(segments):
        color = MODE_COLORS.get(seg["mode"], "#FFFFFF")
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color + "20")

    ax.set_title("Recording Segments Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze real-world deployment recordings.")
    parser.add_argument("json_file", help="Path to the recorded JSON file.")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    print(f"Loaded {data['num_steps']} steps from {data['source']}")
    print(f"  compliance_k={data['compliance_k']}, ema_alpha={data['ema_alpha']}")

    # Find recording segments
    raw_segments = find_segments(data["sim_real_recording"])
    print(f"\nFound {len(raw_segments)} recording segments:")

    segments = []
    for i, (start, end) in enumerate(raw_segments):
        seg = extract_segment(data, start, end)
        segments.append(seg)
        duration = seg["time_s"][-1] - seg["time_s"][0]
        print(f"  Seg {i}: mode={seg['mode']:>8s}, duration={duration:6.1f}s, "
              f"steps=[{start}:{end}]")

    # Generate PDF
    out_dir = os.path.dirname(args.json_file)
    base = os.path.splitext(os.path.basename(args.json_file))[0]
    pdf_path = os.path.join(out_dir, f"{base}_analysis.pdf")

    obs_labels = data.get("obs_labels", [])

    with PdfPages(pdf_path) as pdf:
        plot_summary_table(segments, pdf)
        plot_overview(segments, pdf)
        for i, seg in enumerate(segments):
            plot_segment(seg, i, obs_labels, pdf)

    print(f"\nAnalysis saved to {pdf_path}")


if __name__ == "__main__":
    main()
