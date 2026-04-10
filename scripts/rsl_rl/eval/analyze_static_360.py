"""Offline analysis of static 360 evaluation raw data.

Loads raw_data.json from a static_360 eval run, computes metrics post-hoc,
and generates a PDF report.

Usage:
    python scripts/rsl_rl/eval/analyze_static_360.py \
        data/eval/go2_lowlevel/static_360_2026-04-07_13-34-42
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_peak_displacement(pos_xy, pos_start):
    """Max XY distance from start position."""
    diffs = pos_xy - pos_start
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.max(dists)) if len(dists) > 0 else 0.0


def compute_effective_compliance(vel_xy, force_xy, dt):
    """SAC-Loco Eq.13: C = (1/T_F) * sum((v)^T F / (F^T F))."""
    f = np.array(force_xy)
    f_sq = np.dot(f, f)
    if f_sq < 1e-6 or len(vel_xy) == 0:
        return 0.0
    T = len(vel_xy)
    c = sum(np.dot(v, f) / f_sq for v in vel_xy) / T
    return float(c)


def compute_steady_state_velocity(vel_xy, last_frac=0.25):
    """Mean velocity magnitude over last fraction of force-on phase."""
    if len(vel_xy) == 0:
        return 0.0
    start = max(0, int(len(vel_xy) * (1 - last_frac)))
    tail = vel_xy[start:]
    return float(np.mean(np.linalg.norm(tail, axis=1)))


def compute_mean_displacement(pos_xy, pos_start, last_frac=0.25):
    """Mean displacement from start over last fraction of force-on phase."""
    if len(pos_xy) == 0:
        return 0.0
    start = max(0, int(len(pos_xy) * (1 - last_frac)))
    tail = pos_xy[start:]
    dists = np.linalg.norm(tail - pos_start, axis=1)
    return float(np.mean(dists))


# ── Plotting helpers ─────────────────────────────────────────────────────────


def polar_heatmap(ax, directions_deg, magnitudes, data_2d, title):
    """Polar heatmap: rows=magnitudes, cols=directions."""
    theta = np.deg2rad(directions_deg)
    dtheta = np.deg2rad(36)
    # Build mesh edges
    theta_edges = np.concatenate([theta - dtheta / 2, [theta[-1] + dtheta / 2]])
    if len(magnitudes) > 1:
        dm = np.diff(magnitudes)
        r_edges = np.concatenate([[magnitudes[0] - dm[0] / 2],
                                   magnitudes[:-1] + dm / 2,
                                   [magnitudes[-1] + dm[-1] / 2]])
    else:
        r_edges = np.array([0, magnitudes[0] * 2])
    T, R = np.meshgrid(theta_edges, r_edges)
    mesh = ax.pcolormesh(T, R, data_2d, cmap="RdYlGn", shading="flat")
    ax.set_title(title, fontsize=11, pad=15)
    return mesh


def polar_plot(ax, directions_deg, values_per_dir, title, ylabel, color):
    """Polar plot with mean + std band."""
    theta = np.deg2rad(directions_deg)
    means, stds = [], []
    for d in directions_deg:
        vals = values_per_dir.get(d, [])
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals) if vals else 0.0)
    means = np.array(means)
    stds = np.array(stds)
    # Close the loop
    theta_c = np.concatenate([theta, [theta[0]]])
    means_c = np.concatenate([means, [means[0]]])
    stds_c = np.concatenate([stds, [stds[0]]])
    ax.plot(theta_c, means_c, color=color, linewidth=2)
    ax.fill_between(theta_c, means_c - stds_c, means_c + stds_c, alpha=0.2, color=color)
    ax.set_title(f"{title}\n({ylabel})", fontsize=10, pad=15)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze static 360 eval data.")
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
    magnitudes = sorted([float(m) for m in results.keys()])

    NUM_DIRECTIONS = 10
    DIRECTIONS_DEG = np.linspace(0, 360, NUM_DIRECTIONS, endpoint=False)

    # ── Check if estimator data is available ────────────────────────────
    sample_trial = results[list(results.keys())[0]][list(results[list(results.keys())[0]].keys())[0]][0]
    has_estimator = "force_est" in sample_trial

    # ── Compute metrics per trial ────────────────────────────────────────
    for mag_str in results:
        for deg_str in results[mag_str]:
            for t in results[mag_str][deg_str]:
                vel = np.array(t["vel_xy"])
                pos = np.array(t["pos_xy"])
                pos_start = np.array(t["pos_start"])
                fxy = t["force_xy"]

                if not t["success"] or len(vel) == 0:
                    t["peak_displacement"] = 0.0
                    t["effective_compliance"] = 0.0
                    t["steady_state_vel"] = 0.0
                    t["mean_displacement"] = 0.0
                    t["estimator_mse"] = None
                    t["estimator_mse_per_step"] = None
                    continue

                t["peak_displacement"] = compute_peak_displacement(pos, pos_start)
                t["effective_compliance"] = compute_effective_compliance(vel, fxy, dt)
                t["steady_state_vel"] = compute_steady_state_velocity(vel)
                t["mean_displacement"] = compute_mean_displacement(pos, pos_start)

                if has_estimator:
                    est = np.array(t["force_est"])
                    gt = np.array(fxy)  # [2] — constant across steps
                    # Compare only XY (first 2 dims of estimate)
                    err = est[:, :2] - gt[np.newaxis, :]
                    mse_per_step = np.mean(err ** 2, axis=1)  # [T]
                    ae_per_step = np.linalg.norm(err, axis=1)  # [T] — L2 error per step
                    t["estimator_mse"] = float(np.mean(mse_per_step))
                    t["estimator_mse_per_step"] = mse_per_step
                    t["estimator_mae"] = float(np.mean(ae_per_step))
                    t["estimator_median_ae"] = float(np.median(ae_per_step))
                    t["estimator_rmse"] = float(np.sqrt(np.mean(mse_per_step)))
                    gt_mag = np.linalg.norm(gt)
                    t["estimator_relative_err"] = float(t["estimator_mae"] / gt_mag * 100) if gt_mag > 1e-3 else 0.0
                    t["estimator_ae_per_step"] = ae_per_step
                    # Per-axis MAE
                    t["estimator_mae_x"] = float(np.mean(np.abs(err[:, 0])))
                    t["estimator_mae_y"] = float(np.mean(np.abs(err[:, 1])))
                    # Angular error (degrees)
                    gt_angle = np.arctan2(gt[1], gt[0])
                    est_angles = np.arctan2(est[:, 1], est[:, 0])
                    angle_diff = est_angles - gt_angle
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    t["estimator_angle_err_deg"] = np.abs(angle_diff) * (180.0 / np.pi)
                    t["estimator_angle_err_mean"] = float(np.mean(t["estimator_angle_err_deg"]))
                    t["estimator_angle_err_median"] = float(np.median(t["estimator_angle_err_deg"]))
                else:
                    t["estimator_mse"] = None
                    t["estimator_mse_per_step"] = None
                    t["estimator_mae"] = None
                    t["estimator_median_ae"] = None
                    t["estimator_rmse"] = None
                    t["estimator_relative_err"] = None
                    t["estimator_ae_per_step"] = None
                    t["estimator_mae_x"] = None
                    t["estimator_mae_y"] = None
                    t["estimator_angle_err_deg"] = None
                    t["estimator_angle_err_mean"] = None
                    t["estimator_angle_err_median"] = None

    # ── Print summary to terminal ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Static 360 Analysis — {args.run_dir}")
    print(f"  Magnitudes: {magnitudes} N")
    print(f"  dt: {dt}s")
    print(f"{'=' * 70}\n")

    for mag in magnitudes:
        mag_str = str(float(mag))
        all_trials = []
        for deg in DIRECTIONS_DEG:
            all_trials.extend(results[mag_str][str(float(deg))])
        succ = [t for t in all_trials if t["success"]]
        n_total = len(all_trials)
        n_succ = len(succ)
        if succ:
            line = (f"  {mag:.0f}N: {n_succ}/{n_total} success "
                    f"({100 * n_succ / n_total:.0f}%)"
                    f"  peak_disp={np.mean([t['peak_displacement'] for t in succ]):.3f}m"
                    f"  C={np.mean([t['effective_compliance'] for t in succ]):.4f} s/kg")
            if has_estimator:
                maes = [t['estimator_mae'] for t in succ if t['estimator_mae'] is not None]
                med_aes = [t['estimator_median_ae'] for t in succ if t['estimator_median_ae'] is not None]
                rel_errs = [t['estimator_relative_err'] for t in succ if t['estimator_relative_err'] is not None]
                ang_errs = [t['estimator_angle_err_median'] for t in succ if t['estimator_angle_err_median'] is not None]
                if maes:
                    line += (f"\n         MAE={np.mean(maes):.2f}N"
                             f"  MedianAE={np.mean(med_aes):.2f}N"
                             f"  RelErr={np.mean(rel_errs):.1f}%"
                             f"  AngErr(med)={np.mean(ang_errs):.1f}°")
            print(line)
        else:
            print(f"  {mag:.0f}N: 0/{n_total} success")

    # ── Generate plots ───────────────────────────────────────────────────
    figures = []

    # ── Page 1: Effective compliance vs direction (one line per magnitude) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for mag in magnitudes:
        mag_str = str(float(mag))
        c_means, c_stds = [], []
        for deg in DIRECTIONS_DEG:
            deg_str = str(float(deg))
            vals = [t["effective_compliance"] for t in results[mag_str][deg_str]
                    if t["success"]]
            c_means.append(np.mean(vals) if vals else 0.0)
            c_stds.append(np.std(vals) if vals else 0.0)
        c_means = np.array(c_means)
        c_stds = np.array(c_stds)
        ax.errorbar(DIRECTIONS_DEG, c_means, yerr=c_stds, marker="o", capsize=3,
                     linewidth=1.5, label=f"{mag:.0f}N")

    ax.set_xlabel("Force Direction (deg)")
    ax.set_ylabel("Effective Compliance C (s/kg)")
    ax.set_title("Effective Compliance vs Force Direction", fontsize=13, fontweight="bold")
    ax.set_xticks(DIRECTIONS_DEG)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig)

    # ── Page 3: Peak displacement polar per magnitude ────────────────
    n_mags = len(magnitudes)
    ncols = min(n_mags, 3)
    nrows = math.ceil(n_mags / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             subplot_kw={"projection": "polar"})
    if n_mags == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    fig.suptitle("Peak Displacement by Direction", fontsize=14, fontweight="bold")

    for i, mag in enumerate(magnitudes):
        ax = axes[i // ncols, i % ncols]
        mag_str = str(float(mag))
        vals = {d: [t["peak_displacement"] for t in results[mag_str][str(float(d))]
                     if t["success"]]
                for d in DIRECTIONS_DEG}
        polar_plot(ax, DIRECTIONS_DEG, vals, f"{mag:.0f}N", "meters", "tab:blue")

    for i in range(n_mags, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)
    plt.tight_layout()
    figures.append(fig)

    # ── Page 4: Compliance polar per magnitude ───────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             subplot_kw={"projection": "polar"})
    if n_mags == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    fig.suptitle("Effective Compliance by Direction", fontsize=14, fontweight="bold")

    for i, mag in enumerate(magnitudes):
        ax = axes[i // ncols, i % ncols]
        mag_str = str(float(mag))
        vals = {d: [t["effective_compliance"] for t in results[mag_str][str(float(d))]
                     if t["success"]]
                for d in DIRECTIONS_DEG}
        polar_plot(ax, DIRECTIONS_DEG, vals, f"{mag:.0f}N", "s/kg", "tab:orange")

    for i in range(n_mags, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)
    plt.tight_layout()
    figures.append(fig)

    # ── Page 5: Velocity time series (one per direction, averaged over trials)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"Velocity Magnitude During Force Application ({magnitudes[0]:.0f}N)"
                 if len(magnitudes) == 1
                 else f"Velocity Magnitude During Force Application (largest mag: {magnitudes[-1]:.0f}N)",
                 fontsize=14, fontweight="bold")
    mag_str = str(float(magnitudes[-1]))  # use largest magnitude

    for j, deg in enumerate(DIRECTIONS_DEG):
        ax = axes[j // 5, j % 5]
        deg_str = str(float(deg))
        trials = [t for t in results[mag_str][deg_str] if t["success"]]
        if trials:
            # Pad/truncate to same length
            max_len = max(len(t["vel_xy"]) for t in trials)
            vel_mags = []
            for t in trials:
                v = np.linalg.norm(np.array(t["vel_xy"]), axis=1)
                if len(v) < max_len:
                    v = np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
                vel_mags.append(v)
            vel_mags = np.array(vel_mags)
            time_s = np.arange(max_len) * dt
            mean_v = np.nanmean(vel_mags, axis=0)
            std_v = np.nanstd(vel_mags, axis=0)
            ax.plot(time_s, mean_v, color="tab:red", linewidth=1.5)
            ax.fill_between(time_s, mean_v - std_v, mean_v + std_v,
                            alpha=0.2, color="tab:red")
        ax.set_title(f"{deg:.0f}°", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Vel (m/s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figures.append(fig)

    # ── Estimator quality pages (only if data available) ───────────────
    if has_estimator:
        # ── Page 6a: Estimator RMSE vs direction (one line per magnitude) ──
        fig, ax = plt.subplots(figsize=(10, 5))
        for mag in magnitudes:
            mag_str = str(float(mag))
            rmse_means, rmse_stds = [], []
            for deg in DIRECTIONS_DEG:
                deg_str = str(float(deg))
                mses = [t["estimator_mse"] for t in results[mag_str][deg_str]
                        if t["success"] and t["estimator_mse"] is not None]
                rmses = [m ** 0.5 for m in mses]
                rmse_means.append(np.mean(rmses) if rmses else 0.0)
                rmse_stds.append(np.std(rmses) if rmses else 0.0)
            rmse_means = np.array(rmse_means)
            rmse_stds = np.array(rmse_stds)
            ax.errorbar(DIRECTIONS_DEG, rmse_means, yerr=rmse_stds, marker="o",
                         capsize=3, linewidth=1.5, label=f"{mag:.0f}N")

        ax.set_xlabel("Force Direction (deg)")
        ax.set_ylabel("RMSE (N)")
        ax.set_title("Force Estimator RMSE vs Direction", fontsize=13, fontweight="bold")
        ax.set_xticks(DIRECTIONS_DEG)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures.append(fig)

        # ── Page 6b: Estimator RMSE polar per magnitude ──
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                 subplot_kw={"projection": "polar"})
        if n_mags == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)
        fig.suptitle("Force Estimator RMSE by Direction", fontsize=14, fontweight="bold")

        for i, mag in enumerate(magnitudes):
            ax = axes[i // ncols, i % ncols]
            mag_str = str(float(mag))
            vals = {d: [t["estimator_mse"] ** 0.5
                        for t in results[mag_str][str(float(d))]
                        if t["success"] and t["estimator_mse"] is not None]
                    for d in DIRECTIONS_DEG}
            polar_plot(ax, DIRECTIONS_DEG, vals, f"{mag:.0f}N", "RMSE (N)", "tab:green")

        for i in range(n_mags, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)
        plt.tight_layout()
        figures.append(fig)

        # ── Page 6c: Estimate vs GT time series per direction (one page per magnitude) ──
        for mag in magnitudes:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            mag_str = str(float(mag))
            fig.suptitle(f"Force Estimate vs GT ({mag:.0f}N) — Fx (blue) Fy (red)",
                         fontsize=14, fontweight="bold")

            for j, deg in enumerate(DIRECTIONS_DEG):
                ax = axes[j // 5, j % 5]
                deg_str = str(float(deg))
                trials = [t for t in results[mag_str][deg_str]
                          if t["success"] and "force_est" in t]
                if trials:
                    t0 = trials[0]
                    est = np.array(t0["force_est"])
                    gt = np.array(t0["force_xy"])
                    time_s = np.arange(len(est)) * dt
                    ax.axhline(gt[0], color="tab:blue", linestyle="--", alpha=0.5, label="GT Fx")
                    ax.axhline(gt[1], color="tab:red", linestyle="--", alpha=0.5, label="GT Fy")
                    ax.plot(time_s, est[:, 0], color="tab:blue", linewidth=1, label="Est Fx")
                    ax.plot(time_s, est[:, 1], color="tab:red", linewidth=1, label="Est Fy")
                ax.set_title(f"{deg:.0f}°", fontsize=9)
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Force (N)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if j == 0:
                    ax.legend(fontsize=6)

            plt.tight_layout()
            figures.append(fig)

        # ── Page 6d: Force error time series per direction (one page per magnitude) ──
        for mag in magnitudes:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            mag_str = str(float(mag))
            fig.suptitle(f"Force Estimation Error Over Time ({mag:.0f}N) — "
                         "mean+std across trials",
                         fontsize=14, fontweight="bold")

            for j, deg in enumerate(DIRECTIONS_DEG):
                ax = axes[j // 5, j % 5]
                deg_str = str(float(deg))
                trials = [t for t in results[mag_str][deg_str]
                          if t["success"] and t.get("estimator_ae_per_step") is not None]
                if trials:
                    max_len = max(len(t["estimator_ae_per_step"]) for t in trials)
                    ae_all = []
                    for t in trials:
                        ae = np.array(t["estimator_ae_per_step"])
                        if len(ae) < max_len:
                            ae = np.pad(ae, (0, max_len - len(ae)), constant_values=np.nan)
                        ae_all.append(ae)
                    ae_all = np.array(ae_all)
                    time_s = np.arange(max_len) * dt
                    mean_ae = np.nanmean(ae_all, axis=0)
                    std_ae = np.nanstd(ae_all, axis=0)
                    ax.plot(time_s, mean_ae, color="tab:purple", linewidth=1.5, label="L2 error")
                    ax.fill_between(time_s, mean_ae - std_ae, mean_ae + std_ae,
                                    alpha=0.2, color="tab:purple")
                ax.set_title(f"{deg:.0f}°", fontsize=9)
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Error (N)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig)

        # ── Page 6e: Angular error time series per direction (one page per magnitude) ──
        for mag in magnitudes:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            mag_str = str(float(mag))
            fig.suptitle(f"Angular Error Over Time ({mag:.0f}N) — "
                         "mean+std across trials",
                         fontsize=14, fontweight="bold")

            for j, deg in enumerate(DIRECTIONS_DEG):
                ax = axes[j // 5, j % 5]
                deg_str = str(float(deg))
                trials = [t for t in results[mag_str][deg_str]
                          if t["success"] and t.get("estimator_angle_err_deg") is not None]
                if trials:
                    max_len = max(len(t["estimator_angle_err_deg"]) for t in trials)
                    ang_all = []
                    for t in trials:
                        ang = np.array(t["estimator_angle_err_deg"])
                        if len(ang) < max_len:
                            ang = np.pad(ang, (0, max_len - len(ang)), constant_values=np.nan)
                        ang_all.append(ang)
                    ang_all = np.array(ang_all)
                    time_s = np.arange(max_len) * dt
                    mean_ang = np.nanmean(ang_all, axis=0)
                    std_ang = np.nanstd(ang_all, axis=0)
                    ax.plot(time_s, mean_ang, color="tab:orange", linewidth=1.5, label="Angle err")
                    ax.fill_between(time_s, mean_ang - std_ang, mean_ang + std_ang,
                                    alpha=0.2, color="tab:orange")
                ax.set_title(f"{deg:.0f}°", fontsize=9)
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Angle Error (°)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig)

        # ── Page 6f: MAE / Median AE / Relative Error vs direction ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Force Estimation Metrics vs Direction", fontsize=14, fontweight="bold")

        for metric_idx, (metric_key, metric_label, color) in enumerate([
            ("estimator_mae", "MAE (N)", "tab:blue"),
            ("estimator_median_ae", "Median AE (N)", "tab:green"),
            ("estimator_relative_err", "Relative Error (%)", "tab:red"),
        ]):
            ax = axes[metric_idx]
            for mag in magnitudes:
                mag_str = str(float(mag))
                means, stds = [], []
                for deg in DIRECTIONS_DEG:
                    deg_str = str(float(deg))
                    vals = [t[metric_key] for t in results[mag_str][deg_str]
                            if t["success"] and t[metric_key] is not None]
                    means.append(np.mean(vals) if vals else 0.0)
                    stds.append(np.std(vals) if vals else 0.0)
                means = np.array(means)
                stds = np.array(stds)
                ax.errorbar(DIRECTIONS_DEG, means, yerr=stds, marker="o",
                             capsize=3, linewidth=1.5, label=f"{mag:.0f}N")
            ax.set_xlabel("Force Direction (deg)")
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label, fontsize=12)
            ax.set_xticks(DIRECTIONS_DEG)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

        # ── Page 6g: Per-axis MAE vs direction ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Per-Axis MAE vs Direction", fontsize=14, fontweight="bold")

        for ax_idx, (axis_key, axis_label) in enumerate([
            ("estimator_mae_x", "MAE Fx (N)"),
            ("estimator_mae_y", "MAE Fy (N)"),
        ]):
            ax = axes[ax_idx]
            for mag in magnitudes:
                mag_str = str(float(mag))
                means = []
                for deg in DIRECTIONS_DEG:
                    deg_str = str(float(deg))
                    vals = [t[axis_key] for t in results[mag_str][deg_str]
                            if t["success"] and t[axis_key] is not None]
                    means.append(np.mean(vals) if vals else 0.0)
                ax.plot(DIRECTIONS_DEG, means, marker="o", linewidth=1.5, label=f"{mag:.0f}N")
            ax.set_xlabel("Force Direction (deg)")
            ax.set_ylabel(axis_label)
            ax.set_title(axis_label, fontsize=12)
            ax.set_xticks(DIRECTIONS_DEG)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    # ── Summary table ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 2 + len(magnitudes) * 0.6))
    ax.axis("off")
    headers = ["Mag (N)", "Success %", "Peak Disp (m)", "Compliance (s/kg)",
               "SS Vel (m/s)", "Mean Disp (m)"]
    if has_estimator:
        headers.extend(["MAE (N)", "Median AE (N)", "RMSE (N)", "Rel Err %",
                         "MAE x (N)", "MAE y (N)", "Ang Err med (°)"])
    rows = []
    for mag in magnitudes:
        mag_str = str(float(mag))
        all_trials = []
        for deg in DIRECTIONS_DEG:
            all_trials.extend(results[mag_str][str(float(deg))])
        succ = [t for t in all_trials if t["success"]]
        if succ:
            row = [
                f"{mag:.0f}",
                f"{np.mean([t['success'] for t in all_trials]) * 100:.0f}%",
                f"{np.mean([t['peak_displacement'] for t in succ]):.3f} ± "
                f"{np.std([t['peak_displacement'] for t in succ]):.3f}",
                f"{np.mean([t['effective_compliance'] for t in succ]):.4f} ± "
                f"{np.std([t['effective_compliance'] for t in succ]):.4f}",
                f"{np.mean([t['steady_state_vel'] for t in succ]):.3f} ± "
                f"{np.std([t['steady_state_vel'] for t in succ]):.3f}",
                f"{np.mean([t['mean_displacement'] for t in succ]):.3f} ± "
                f"{np.std([t['mean_displacement'] for t in succ]):.3f}",
            ]
            if has_estimator:
                def _avg(key):
                    vals = [t[key] for t in succ if t[key] is not None]
                    return np.mean(vals) if vals else float("nan")
                row.extend([
                    f"{_avg('estimator_mae'):.2f}",
                    f"{_avg('estimator_median_ae'):.2f}",
                    f"{_avg('estimator_rmse'):.2f}",
                    f"{_avg('estimator_relative_err'):.1f}",
                    f"{_avg('estimator_mae_x'):.2f}",
                    f"{_avg('estimator_mae_y'):.2f}",
                    f"{_avg('estimator_angle_err_median'):.1f}",
                ])
            rows.append(row)
        else:
            row = [f"{mag:.0f}", "0%", "N/A", "N/A", "N/A", "N/A"]
            if has_estimator:
                row.extend(["N/A"] * 7)
            rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    fig.suptitle("Static 360 Evaluation Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    figures.append(fig)

    # ── Save PDF ─────────────────────────────────────────────────────
    from matplotlib.backends.backend_pdf import PdfPages
    out_path = os.path.join(args.run_dir, "analysis.pdf")
    with PdfPages(out_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n  Analysis saved to {out_path}")


if __name__ == "__main__":
    main()
