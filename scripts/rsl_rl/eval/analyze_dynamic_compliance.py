"""Offline analysis of dynamic compliance evaluation raw data.

Loads raw_data.json from a dynamic_compliance eval run, computes metrics,
and generates analysis.pdf.

Usage:
    python scripts/rsl_rl/eval/analyze_dynamic_compliance.py \
        data/eval/go2_lowlevel/dynamic_compliance_2026-04-07_XX-XX-XX
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_decay_time(lat_vel, dt, threshold_frac=0.1):
    """Time for lateral velocity to drop below threshold_frac * peak."""
    if len(lat_vel) == 0:
        return 0.0
    peak = np.max(np.abs(lat_vel))
    if peak < 1e-6:
        return 0.0
    threshold = threshold_frac * peak
    peak_idx = np.argmax(np.abs(lat_vel))
    after_peak = np.abs(lat_vel[peak_idx:])
    below = np.where(after_peak < threshold)[0]
    if len(below) > 0:
        return below[0] * dt
    return len(after_peak) * dt


def compute_effective_compliance(vel_xy, cmd_vel_xy, force_xy, dt):
    """SAC-Loco Eq.13: C = (1/T) * sum((v-v')^T F / (F^T F))."""
    f = np.array(force_xy)
    f_sq = np.dot(f, f)
    if f_sq < 1e-6 or len(vel_xy) == 0:
        return 0.0
    T = len(vel_xy)
    delta_v = vel_xy - cmd_vel_xy
    return float(np.sum(np.dot(delta_v, f)) / (f_sq * T))


def compute_peak_lateral_dev(pos_xy, y_start):
    """Max absolute lateral deviation from desired path."""
    if len(pos_xy) == 0:
        return 0.0
    return float(np.max(np.abs(pos_xy[:, 1] - y_start)))


def compute_return_to_path_time(recovered_step, dt):
    """Time to return to path from recovery start."""
    return float(recovered_step * dt)


def compute_estimator_mse(force_est, force_xy):
    """MSE of force estimate vs GT (XY only)."""
    if force_est is None or len(force_est) == 0:
        return None
    est = np.array(force_est)
    gt = np.array(force_xy)
    err = est[:, :2] - gt[np.newaxis, :]
    return float(np.mean(err ** 2))


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze dynamic compliance eval data.")
    parser.add_argument("run_dir", type=str, help="Path to eval run directory.")
    args = parser.parse_args()

    raw_path = os.path.join(args.run_dir, "raw_data.json")
    config_path = os.path.join(args.run_dir, "config.json")

    with open(raw_path) as f:
        data = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    results = data["results"] if "results" in data else data
    dt = config["dt"]
    walk_speed = config.get("walk_speed", 0.5)
    magnitudes = sorted([float(m) for m in results.keys()])

    # ── Compute metrics per cycle per trial ──────────────────────────────
    for mag_str in results:
        mag = float(mag_str)
        for trial in results[mag_str]:
            y_start = trial["y_start"]
            for cyc in trial["cycles"]:
                direction = cyc["direction"]
                force_sign = 1.0 if direction == "left" else -1.0
                fy = force_sign * mag
                force_xy = [0.0, fy]

                fo_vel = np.array(cyc["force_on"]["vel_xy"])
                fo_pos = np.array(cyc["force_on"]["pos_xy"])
                rec_vel = np.array(cyc["recovery"]["vel_xy"])
                rec_pos = np.array(cyc["recovery"]["pos_xy"])

                # Decay time from recovery lateral velocity
                if len(rec_vel) > 0:
                    cyc["decay_time"] = compute_decay_time(rec_vel[:, 1], dt)
                else:
                    cyc["decay_time"] = config.get("recovery_timeout_s", 10.0)

                # Effective compliance during force-on
                if len(fo_vel) > 0:
                    cmd = np.tile([walk_speed, 0.0], (len(fo_vel), 1))
                    cyc["effective_compliance"] = compute_effective_compliance(
                        fo_vel, cmd, force_xy, dt
                    )
                else:
                    cyc["effective_compliance"] = 0.0

                # Peak lateral deviation (force-on + recovery combined)
                all_pos = []
                if len(fo_pos) > 0:
                    all_pos.append(fo_pos)
                if len(rec_pos) > 0:
                    all_pos.append(rec_pos)
                if all_pos:
                    combined = np.vstack(all_pos)
                    cyc["peak_lateral_dev"] = compute_peak_lateral_dev(combined, y_start)
                else:
                    cyc["peak_lateral_dev"] = 0.0

                # Return-to-path time
                cyc["return_to_path_time"] = compute_return_to_path_time(
                    cyc["recovered_step"], dt
                )

                # Velocity offset during force-on
                if len(fo_vel) > 0:
                    cmd = np.tile([walk_speed, 0.0], (len(fo_vel), 1))
                    cyc["velocity_offset"] = float(np.mean(np.linalg.norm(fo_vel - cmd, axis=1)))
                else:
                    cyc["velocity_offset"] = 0.0

                # Estimator MSE
                force_est = cyc["force_on"].get("force_est")
                cyc["estimator_mse"] = compute_estimator_mse(force_est, force_xy)

    # ── Helper: collect all cycles across successful trials ──────────────
    def all_cycles(mag_str):
        cycles = []
        for trial in results[mag_str]:
            if trial["success"]:
                cycles.extend(trial["cycles"])
        return cycles

    # ── Print summary ────────────────────────────────────────────────────
    has_estimator = any(
        cyc.get("estimator_mse") is not None
        for mag_str in results
        for trial in results[mag_str]
        for cyc in trial["cycles"]
    )

    print(f"\n{'=' * 70}")
    print(f"  Dynamic Compliance Analysis — {args.run_dir}")
    print(f"  Magnitudes: {magnitudes} N, Walk speed: {walk_speed} m/s")
    print(f"{'=' * 70}\n")

    for mag in magnitudes:
        mag_str = str(float(mag))
        trials = results[mag_str]
        cycles = all_cycles(mag_str)
        n_succ = sum(t["success"] for t in trials)
        print(f"  {mag:.0f}N: {n_succ}/{len(trials)} trials success, {len(cycles)} cycles")
        if cycles:
            print(f"    decay={np.mean([c['decay_time'] for c in cycles]):.2f}s"
                  f"  dev={np.mean([c['peak_lateral_dev'] for c in cycles]):.3f}m"
                  f"  C={np.mean([c['effective_compliance'] for c in cycles]):.4f}"
                  f"  return={np.mean([c['return_to_path_time'] for c in cycles]):.2f}s")
            if has_estimator:
                mses = [c["estimator_mse"] for c in cycles if c["estimator_mse"] is not None]
                if mses:
                    print(f"    est RMSE={np.mean(mses)**0.5:.3f}N")

    # ── Generate plots ───────────────────────────────────────────────────
    figures = []
    n_mags = len(magnitudes)

    # ── Page 1: Lateral velocity decay after force release ───────────
    fig, axes = plt.subplots(1, n_mags, figsize=(5 * n_mags, 5), squeeze=False)
    fig.suptitle("Lateral Velocity Decay After Force Release", fontsize=14, fontweight="bold")

    for i, mag in enumerate(magnitudes):
        ax = axes[0, i]
        cycles = all_cycles(str(float(mag)))
        if cycles:
            traces = []
            max_len = max((len(np.array(c["recovery"]["vel_xy"])) for c in cycles
                           if len(c["recovery"]["vel_xy"]) > 0), default=0)
            if max_len > 0:
                for c in cycles:
                    rv = np.array(c["recovery"]["vel_xy"])
                    if len(rv) == 0:
                        continue
                    lat_v = np.abs(rv[:, 1])
                    padded = np.full(max_len, np.nan)
                    padded[:len(lat_v)] = lat_v
                    traces.append(padded)
                if traces:
                    traces = np.array(traces)
                    t_axis = np.arange(max_len) * dt
                    ax.plot(t_axis, np.nanmean(traces, axis=0),
                            color="tab:purple", linewidth=2, label="Mean")
                    ax.fill_between(t_axis, np.nanmin(traces, axis=0),
                                    np.nanmax(traces, axis=0),
                                    alpha=0.2, color="tab:purple", label="Min/Max")
                    ax.legend(fontsize=8)
        ax.set_title(f"{mag:.0f}N", fontsize=12)
        ax.set_xlabel("Time after release (s)")
        if i == 0:
            ax.set_ylabel("Lateral velocity (m/s)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig)

    # ── Page 2: Lateral deviation time series ────────────────────────
    fig, axes = plt.subplots(1, n_mags, figsize=(5 * n_mags, 5), squeeze=False)
    fig.suptitle("Lateral Deviation During Force + Recovery", fontsize=14, fontweight="bold")

    for i, mag in enumerate(magnitudes):
        ax = axes[0, i]
        cycles = all_cycles(str(float(mag)))
        for c in cycles:
            fo_pos = np.array(c["force_on"]["pos_xy"])
            rec_pos = np.array(c["recovery"]["pos_xy"])
            if len(fo_pos) == 0:
                continue
            y_ref = fo_pos[0, 1]
            color = "tab:blue" if c["direction"] == "left" else "tab:red"
            fo_t = np.arange(len(fo_pos)) * dt
            ax.plot(fo_t, fo_pos[:, 1] - y_ref, color=color, alpha=0.3, linewidth=0.8)
            if len(rec_pos) > 0:
                rec_t = (len(fo_pos) + np.arange(len(rec_pos))) * dt
                ax.plot(rec_t, rec_pos[:, 1] - y_ref, color=color, alpha=0.3,
                        linewidth=0.8, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{mag:.0f}N", fontsize=12)
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_ylabel("Lateral deviation (m)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig)

    # ── Page 3: Metrics vs magnitude (4 subplots) ───────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Dynamic Compliance Metrics vs Force Magnitude", fontsize=14, fontweight="bold")

    metric_cfgs = [
        ("decay_time", "Decay Time (s)", "tab:purple"),
        ("effective_compliance", "Compliance C (s/kg)", "tab:blue"),
        ("peak_lateral_dev", "Peak Lateral Dev (m)", "tab:red"),
        ("return_to_path_time", "Return-to-Path Time (s)", "tab:green"),
    ]
    for idx, (key, ylabel, color) in enumerate(metric_cfgs):
        ax = axes[idx // 2, idx % 2]
        means, stds = [], []
        for mag in magnitudes:
            cycles = all_cycles(str(float(mag)))
            vals = [c[key] for c in cycles]
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)
        ax.errorbar(magnitudes, means, yerr=stds, marker="o", capsize=4,
                     linewidth=2, color=color)
        ax.set_xlabel("Force Magnitude (N)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig)

    # ── Page 4: Velocity offset bar chart ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    v_means, v_stds = [], []
    for mag in magnitudes:
        cycles = all_cycles(str(float(mag)))
        vals = [c["velocity_offset"] for c in cycles]
        v_means.append(np.mean(vals) if vals else 0.0)
        v_stds.append(np.std(vals) if vals else 0.0)
    ax.bar(range(n_mags), v_means, yerr=v_stds, capsize=4, color="tab:orange", alpha=0.8)
    ax.set_xticks(range(n_mags))
    ax.set_xticklabels([f"{m:.0f}N" for m in magnitudes])
    ax.set_xlabel("Force Magnitude")
    ax.set_ylabel("Velocity Offset (m/s)")
    ax.set_title("Mean Velocity Offset During Force Application", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    figures.append(fig)

    # ── Estimator pages (if available) ───────────────────────────────
    if has_estimator:
        # RMSE vs magnitude
        fig, ax = plt.subplots(figsize=(8, 5))
        rmse_means, rmse_stds = [], []
        for mag in magnitudes:
            cycles = all_cycles(str(float(mag)))
            mses = [c["estimator_mse"] for c in cycles if c["estimator_mse"] is not None]
            rmses = [m ** 0.5 for m in mses]
            rmse_means.append(np.mean(rmses) if rmses else 0.0)
            rmse_stds.append(np.std(rmses) if rmses else 0.0)
        ax.errorbar(magnitudes, rmse_means, yerr=rmse_stds, marker="o", capsize=4,
                     linewidth=2, color="tab:green")
        ax.set_xlabel("Force Magnitude (N)")
        ax.set_ylabel("RMSE (N)")
        ax.set_title("Force Estimator RMSE vs Magnitude (During Walking)", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures.append(fig)

        # Estimate vs GT time series for first cycle of largest magnitude
        mag_str = str(float(magnitudes[-1]))
        trials = [t for t in results[mag_str] if t["success"]]
        if trials and trials[0]["cycles"]:
            cyc = trials[0]["cycles"][0]
            force_est = cyc["force_on"].get("force_est")
            if force_est is not None:
                est = np.array(force_est)
                mag = magnitudes[-1]
                direction = cyc["direction"]
                fy = mag if direction == "left" else -mag
                gt = [0.0, fy]
                time_s = np.arange(len(est)) * dt

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.axhline(gt[0], color="tab:blue", linestyle="--", alpha=0.5, label="GT Fx")
                ax.axhline(gt[1], color="tab:red", linestyle="--", alpha=0.5, label="GT Fy")
                ax.plot(time_s, est[:, 0], color="tab:blue", linewidth=1.5, label="Est Fx")
                ax.plot(time_s, est[:, 1], color="tab:red", linewidth=1.5, label="Est Fy")
                if est.shape[1] > 2:
                    ax.plot(time_s, est[:, 2], color="tab:green", linewidth=1.5, label="Est Fz")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Force (N)")
                ax.set_title(f"Force Estimate vs GT — {magnitudes[-1]:.0f}N {direction}",
                             fontsize=13, fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                figures.append(fig)

    # ── Summary table ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 2 + len(magnitudes) * 0.6))
    ax.axis("off")
    headers = ["Mag (N)", "Success %", "Decay (s)", "Compliance (s/kg)",
               "Peak Dev (m)", "Return (s)", "Vel Offset (m/s)"]
    if has_estimator:
        headers.append("Est RMSE (N)")
    rows = []
    for mag in magnitudes:
        mag_str = str(float(mag))
        trials = results[mag_str]
        cycles = all_cycles(mag_str)
        succ_rate = np.mean([t["success"] for t in trials]) * 100 if trials else 0

        def fmt(key):
            vals = [c[key] for c in cycles]
            if not vals:
                return "N/A"
            return f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}"

        row = [f"{mag:.0f}", f"{succ_rate:.0f}%", fmt("decay_time"),
               fmt("effective_compliance"), fmt("peak_lateral_dev"),
               fmt("return_to_path_time"), fmt("velocity_offset")]
        if has_estimator:
            mses = [c["estimator_mse"] for c in cycles if c["estimator_mse"] is not None]
            row.append(f"{np.mean(mses)**0.5:.3f}" if mses else "N/A")
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    fig.suptitle("Dynamic Compliance Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    figures.append(fig)

    # ── Save PDF ─────────────────────────────────────────────────────
    out_path = os.path.join(args.run_dir, "analysis.pdf")
    with PdfPages(out_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n  Analysis saved to {out_path}")


if __name__ == "__main__":
    main()
