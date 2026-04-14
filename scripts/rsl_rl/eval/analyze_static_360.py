"""Offline analysis of static 360 evaluation raw data.

Supports both v1 (force-only, XY plane) and v2 (elevation + torque sweep) formats.
Auto-detects force_dim and generates appropriate plots.

Usage:
    python scripts/rsl_rl/eval/analyze_static_360.py \
        data/eval/go2_lowlevel/static_360_2026-04-07_13-34-42
    python scripts/rsl_rl/eval/analyze_static_360.py \
        data/eval/ablation_S6/static_360_... --task Go2-Ablation-S6-v0
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
    diffs = pos_xy - pos_start
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.max(dists)) if len(dists) > 0 else 0.0


def compute_mean_displacement(pos_xy, pos_start, last_frac=0.25):
    if len(pos_xy) == 0:
        return 0.0
    start = max(0, int(len(pos_xy) * (1 - last_frac)))
    tail = pos_xy[start:]
    dists = np.linalg.norm(tail - pos_start, axis=1)
    return float(np.mean(dists))


def _build_gt_vector(trial, force_dim, force_layout, is_torque=False):
    """Build GT vector matching the estimator's output layout."""
    if is_torque:
        tq_vec = np.array(trial["torque_vec"])  # [3]: roll, pitch, yaw
        if force_layout == "xy_yaw":
            return np.array([0.0, 0.0, tq_vec[2]])
        if force_dim == 2:
            return np.array([0.0, 0.0])
        if force_dim == 3:
            return np.array([0.0, 0.0, 0.0])
        if force_dim == 4:
            return np.array([0.0, 0.0, 0.0, tq_vec[2]])
        return np.concatenate([np.array([0.0, 0.0, 0.0]), tq_vec])[:force_dim]

    fxyz = np.array(trial.get("force_xyz", list(trial["force_xy"]) + [0.0]))
    if force_layout == "xy_yaw":
        return np.array([fxyz[0], fxyz[1], 0.0])
    if force_dim <= 3:
        return fxyz[:force_dim]
    if force_dim == 4:
        return np.array([fxyz[0], fxyz[1], fxyz[2], 0.0])
    return np.concatenate([fxyz[:3], np.array([0.0, 0.0, 0.0])])[:force_dim]


DIM_LABELS = {
    2: ["Fx", "Fy"],
    3: ["Fx", "Fy", "Fz"],
    4: ["Fx", "Fy", "Fz", "τ_yaw"],
    6: ["Fx", "Fy", "Fz", "τ_roll", "τ_pitch", "τ_yaw"],
}
DIM_LABELS_XY_YAW = ["Fx", "Fy", "τ_yaw"]


def _get_dim_labels(force_dim, force_layout):
    if force_layout == "xy_yaw":
        return DIM_LABELS_XY_YAW
    return DIM_LABELS.get(force_dim, [f"d{i}" for i in range(force_dim)])


# ── Plotting helpers ─────────────────────────────────────────────────────────


def polar_plot(ax, directions_deg, values_per_dir, title, ylabel, color):
    theta = np.deg2rad(directions_deg)
    means, stds = [], []
    for d in directions_deg:
        vals = values_per_dir.get(d, [])
        means.append(np.mean(vals) if vals else 0.0)
        stds.append(np.std(vals) if vals else 0.0)
    means = np.array(means)
    stds = np.array(stds)
    theta_c = np.concatenate([theta, [theta[0]]])
    means_c = np.concatenate([means, [means[0]]])
    stds_c = np.concatenate([stds, [stds[0]]])
    ax.plot(theta_c, means_c, color=color, linewidth=2)
    ax.fill_between(theta_c, means_c - stds_c, means_c + stds_c, alpha=0.2, color=color)
    ax.set_title(f"{title}\n({ylabel})", fontsize=10, pad=15)


# ── Data access helpers ──────────────────────────────────────────────────────


def _iter_force_trials(force_results, elevation_angles, is_v2):
    """Yield (mag_str, deg_str, elev_str, trials_list) across the data."""
    for mag_str in force_results:
        for deg_str in force_results[mag_str]:
            if is_v2:
                for elev_str in force_results[mag_str][deg_str]:
                    yield mag_str, deg_str, elev_str, force_results[mag_str][deg_str][elev_str]
            else:
                yield mag_str, deg_str, "0.0", force_results[mag_str][deg_str]


def _get_trials(force_results, mag_str, deg_str, elev_str, is_v2):
    if is_v2:
        return force_results.get(mag_str, {}).get(deg_str, {}).get(elev_str, [])
    return force_results.get(mag_str, {}).get(deg_str, [])


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze static 360 eval data.")
    parser.add_argument("run_dir", type=str, help="Path to eval run directory.")
    parser.add_argument("--task", type=str, default=None, help="Task name for labeling.")
    args = parser.parse_args()

    raw_path = os.path.join(args.run_dir, "raw_data.json")
    config_path = os.path.join(args.run_dir, "config.json")

    with open(raw_path) as f:
        data = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    dt = config["dt"]
    task_name = args.task or config.get("task", "unknown")

    # ── Format detection ────────────────────────────────────────────────
    is_v2 = "force_sweep" in data
    if is_v2:
        meta = data["metadata"]
        force_results = data["force_sweep"]
        torque_results = data.get("torque_sweep")
        force_dim = meta["force_dim"]
        force_layout = meta.get("force_layout", "auto")
        elevation_angles = meta.get("elevation_angles_deg", [0])
    else:
        force_results = data["results"] if "results" in data else data
        torque_results = None
        elevation_angles = [0]
        # Infer force_dim from data
        sample_mag = list(force_results.keys())[0]
        sample_deg = list(force_results[sample_mag].keys())[0]
        sample_trial = force_results[sample_mag][sample_deg][0]
        if "force_est" in sample_trial:
            force_dim = len(sample_trial["force_est"][0])
        else:
            force_dim = 2
        force_layout = "auto"

    has_elevation = len(elevation_angles) > 1
    has_torque = torque_results is not None and len(torque_results) > 0
    dim_labels = _get_dim_labels(force_dim, force_layout)

    magnitudes = sorted([float(m) for m in force_results.keys()])
    NUM_DIRECTIONS = 10
    DIRECTIONS_DEG = np.linspace(0, 360, NUM_DIRECTIONS, endpoint=False)

    # ── Check estimator availability ────────────────────────────────────
    sample_trial = None
    for _, _, _, trials in _iter_force_trials(force_results, elevation_angles, is_v2):
        if trials:
            sample_trial = trials[0]
            break
    has_estimator = sample_trial is not None and "force_est" in sample_trial

    # ── Compute metrics per force trial ─────────────────────────────────
    for mag_str, deg_str, elev_str, trials in _iter_force_trials(force_results, elevation_angles, is_v2):
        for t in trials:
            vel = np.array(t["vel_xy"])
            pos = np.array(t["pos_xy"])
            pos_start = np.array(t["pos_start"])

            if not t["success"] or len(vel) == 0:
                t["peak_displacement"] = 0.0
                t["mean_displacement"] = 0.0
                t["estimator_metrics"] = None
                continue

            t["peak_displacement"] = compute_peak_displacement(pos, pos_start)
            t["mean_displacement"] = compute_mean_displacement(pos, pos_start)

            if has_estimator:
                est = np.array(t["force_est"])
                gt = _build_gt_vector(t, force_dim, force_layout, is_torque=False)
                err = est - gt[np.newaxis, :]
                ae_per_step = np.linalg.norm(err, axis=1)
                gt_mag = np.linalg.norm(gt)

                per_axis_mae = [float(np.mean(np.abs(err[:, d]))) for d in range(force_dim)]
                angle_err_mean = None
                if gt_mag > 1e-3 and force_dim >= 2:
                    gt_angle = np.arctan2(gt[1], gt[0])
                    est_angles = np.arctan2(est[:, 1], est[:, 0])
                    angle_diff = np.arctan2(np.sin(est_angles - gt_angle), np.cos(est_angles - gt_angle))
                    angle_err_mean = float(np.median(np.abs(angle_diff) * 180.0 / np.pi))

                t["estimator_metrics"] = {
                    "mae": float(np.mean(ae_per_step)),
                    "median_ae": float(np.median(ae_per_step)),
                    "relative_err": float(np.mean(ae_per_step) / gt_mag * 100) if gt_mag > 1e-3 else 0.0,
                    "per_axis_mae": per_axis_mae,
                    "angle_err_median": angle_err_mean,
                    "ae_per_step": ae_per_step,
                }
            else:
                t["estimator_metrics"] = None

    # ── Compute metrics per torque trial ────────────────────────────────
    if has_torque and has_estimator:
        for axis_name in torque_results:
            for mag_str in torque_results[axis_name]:
                for sign_str in torque_results[axis_name][mag_str]:
                    for t in torque_results[axis_name][mag_str][sign_str]:
                        if not t["success"]:
                            t["estimator_metrics"] = None
                            continue
                        if "force_est" not in t:
                            t["estimator_metrics"] = None
                            continue
                        est = np.array(t["force_est"])
                        gt = _build_gt_vector(t, force_dim, force_layout, is_torque=True)
                        err = est - gt[np.newaxis, :]
                        ae_per_step = np.linalg.norm(err, axis=1)
                        per_axis_mae = [float(np.mean(np.abs(err[:, d]))) for d in range(force_dim)]
                        per_axis_bias = [float(np.mean(err[:, d])) for d in range(force_dim)]
                        t["estimator_metrics"] = {
                            "mae": float(np.mean(ae_per_step)),
                            "median_ae": float(np.median(ae_per_step)),
                            "per_axis_mae": per_axis_mae,
                            "per_axis_bias": per_axis_bias,
                            "ae_per_step": ae_per_step,
                        }

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Static 360 Analysis — {args.run_dir}")
    print(f"  Task: {task_name}")
    print(f"  Force dim: {force_dim}  Layout: {force_layout}")
    print(f"  Magnitudes: {magnitudes} N")
    print(f"  Elevations: {elevation_angles} deg")
    print(f"  Torque sweep: {has_torque}")
    print(f"{'=' * 70}\n")

    for mag in magnitudes:
        mag_str = str(float(mag))
        all_trials = []
        for deg in DIRECTIONS_DEG:
            deg_str = str(float(deg))
            for elev in elevation_angles:
                all_trials.extend(_get_trials(force_results, mag_str, deg_str, str(float(elev)), is_v2))
        succ = [t for t in all_trials if t["success"]]
        n_total = len(all_trials)
        n_succ = len(succ)
        line = f"  {mag:.0f}N: {n_succ}/{n_total} success ({100 * n_succ / n_total:.0f}%)"
        if succ:
            line += f"  peak_disp={np.mean([t['peak_displacement'] for t in succ]):.3f}m"
            if has_estimator:
                maes = [t['estimator_metrics']['mae'] for t in succ if t['estimator_metrics']]
                if maes:
                    line += f"  MAE={np.mean(maes):.2f}N"
        print(line)

    # ═══════════════════════════════════════════════════════════════════
    # Generate plots
    # ═══════════════════════════════════════════════════════════════════
    figures = []
    elev0_str = str(float(elevation_angles[0]))

    # ── Page 1: Peak displacement polar per magnitude (elev=0) ──────
    n_mags = len(magnitudes)
    ncols = min(n_mags, 3)
    nrows = math.ceil(n_mags / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             subplot_kw={"projection": "polar"})
    if n_mags == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Peak Displacement by Direction (elev=0°)\n{task_name}", fontsize=14, fontweight="bold")
    for i, mag in enumerate(magnitudes):
        ax = axes[i // ncols, i % ncols]
        mag_str = str(float(mag))
        vals = {d: [t["peak_displacement"] for t in _get_trials(force_results, mag_str, str(float(d)), elev0_str, is_v2) if t["success"]] for d in DIRECTIONS_DEG}
        polar_plot(ax, DIRECTIONS_DEG, vals, f"{mag:.0f}N", "meters", "tab:blue")
    for i in range(n_mags, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)
    plt.tight_layout()
    figures.append(fig)

    # ── Estimator quality pages (force sweep, elev=0) ───────────────
    if has_estimator:
        force_labels = [l for l in dim_labels if l.startswith("F")]
        torque_labels = [l for l in dim_labels if l.startswith("τ") or l.startswith("t")]
        force_indices = [i for i, l in enumerate(dim_labels) if l.startswith("F")]
        torque_indices = [i for i, l in enumerate(dim_labels) if l.startswith("τ") or l.startswith("t")]
        force_colors = ["tab:blue", "tab:red", "tab:green"]
        torque_colors = ["tab:purple", "tab:brown", "tab:orange"]

        # Force estimate vs GT (Fx, Fy, Fz only) per direction, one page per magnitude
        for mag in magnitudes:
            mag_str = str(float(mag))
            fig, axes_grid = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f"Force Estimate vs GT ({mag:.0f}N, elev=0°) — {task_name}\n"
                         f"{' / '.join(force_labels)}",
                         fontsize=14, fontweight="bold")
            for j, deg in enumerate(DIRECTIONS_DEG):
                ax = axes_grid[j // 5, j % 5]
                deg_str = str(float(deg))
                trials = [t for t in _get_trials(force_results, mag_str, deg_str, elev0_str, is_v2)
                          if t["success"] and "force_est" in t]
                if trials:
                    t0 = trials[0]
                    est = np.array(t0["force_est"])
                    gt = _build_gt_vector(t0, force_dim, force_layout)
                    time_s = np.arange(len(est)) * dt
                    for ci, d in enumerate(force_indices):
                        ax.axhline(gt[d], color=force_colors[ci % len(force_colors)],
                                   linestyle="--", alpha=0.4)
                        ax.plot(time_s, est[:, d], color=force_colors[ci % len(force_colors)],
                                linewidth=0.8, label=dim_labels[d] if j == 0 else None)
                ax.set_title(f"{deg:.0f}°", fontsize=9)
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Force (N)", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                if j == 0:
                    ax.legend(fontsize=6)
            plt.tight_layout()
            figures.append(fig)

        # Torque estimate vs GT (τ only) per direction, one page per magnitude
        if torque_indices:
            for mag in magnitudes:
                mag_str = str(float(mag))
                fig, axes_grid = plt.subplots(2, 5, figsize=(20, 8))
                fig.suptitle(f"Torque Estimate vs GT ({mag:.0f}N, elev=0°) — {task_name}\n"
                             f"{' / '.join(torque_labels)}",
                             fontsize=14, fontweight="bold")
                for j, deg in enumerate(DIRECTIONS_DEG):
                    ax = axes_grid[j // 5, j % 5]
                    deg_str = str(float(deg))
                    trials = [t for t in _get_trials(force_results, mag_str, deg_str, elev0_str, is_v2)
                              if t["success"] and "force_est" in t]
                    if trials:
                        t0 = trials[0]
                        est = np.array(t0["force_est"])
                        gt = _build_gt_vector(t0, force_dim, force_layout)
                        time_s = np.arange(len(est)) * dt
                        for ci, d in enumerate(torque_indices):
                            ax.axhline(gt[d], color=torque_colors[ci % len(torque_colors)],
                                       linestyle="--", alpha=0.4)
                            ax.plot(time_s, est[:, d], color=torque_colors[ci % len(torque_colors)],
                                    linewidth=0.8, label=dim_labels[d] if j == 0 else None)
                    ax.set_title(f"{deg:.0f}°", fontsize=9)
                    ax.set_xlabel("Time (s)", fontsize=8)
                    ax.set_ylabel("Torque (Nm)", fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.3)
                    if j == 0:
                        ax.legend(fontsize=6)
                plt.tight_layout()
                figures.append(fig)

        # MAE / Median AE / Relative Error vs direction (elev=0)
        fig, axes_row = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Force Estimation Metrics vs Direction (elev=0°) — {task_name}",
                     fontsize=14, fontweight="bold")
        for metric_idx, (metric_key, metric_label, color) in enumerate([
            ("mae", "MAE (N)", "tab:blue"),
            ("median_ae", "Median AE (N)", "tab:green"),
            ("relative_err", "Relative Error (%)", "tab:red"),
        ]):
            ax = axes_row[metric_idx]
            for mag in magnitudes:
                mag_str = str(float(mag))
                means, stds = [], []
                for deg in DIRECTIONS_DEG:
                    deg_str = str(float(deg))
                    vals = [t["estimator_metrics"][metric_key]
                            for t in _get_trials(force_results, mag_str, deg_str, elev0_str, is_v2)
                            if t["success"] and t.get("estimator_metrics") and metric_key in t["estimator_metrics"]]
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

        # Per-axis MAE vs direction (elev=0)
        n_ax = min(force_dim, 6)
        ncols_ax = min(n_ax, 3)
        nrows_ax = math.ceil(n_ax / ncols_ax)
        fig, axes_grid = plt.subplots(nrows_ax, ncols_ax, figsize=(6 * ncols_ax, 4.5 * nrows_ax))
        if n_ax == 1:
            axes_grid = np.array([[axes_grid]])
        axes_grid = np.atleast_2d(axes_grid)
        fig.suptitle(f"Per-Axis MAE vs Direction (elev=0°) — {task_name}", fontsize=14, fontweight="bold")
        for d_idx in range(n_ax):
            ax = axes_grid[d_idx // ncols_ax, d_idx % ncols_ax]
            for mag in magnitudes:
                mag_str = str(float(mag))
                means = []
                for deg in DIRECTIONS_DEG:
                    deg_str = str(float(deg))
                    vals = [t["estimator_metrics"]["per_axis_mae"][d_idx]
                            for t in _get_trials(force_results, mag_str, deg_str, elev0_str, is_v2)
                            if t["success"] and t.get("estimator_metrics")]
                    means.append(np.mean(vals) if vals else 0.0)
                ax.plot(DIRECTIONS_DEG, means, marker="o", linewidth=1.5, label=f"{mag:.0f}N")
            ax.set_xlabel("Force Direction (deg)")
            ax.set_ylabel(f"MAE {dim_labels[d_idx]} (N)")
            ax.set_title(f"MAE {dim_labels[d_idx]}", fontsize=12)
            ax.set_xticks(DIRECTIONS_DEG)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        for i in range(n_ax, nrows_ax * ncols_ax):
            axes_grid[i // ncols_ax, i % ncols_ax].set_visible(False)
        plt.tight_layout()
        figures.append(fig)

    # ── Elevation pages (when has_elevation) ────────────────────────
    if has_elevation and has_estimator:
        # MAE vs elevation angle (one line per magnitude)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"MAE vs Elevation Angle — {task_name}", fontsize=13, fontweight="bold")
        for mag in magnitudes:
            mag_str = str(float(mag))
            elev_maes = []
            for elev in elevation_angles:
                elev_str = str(float(elev))
                all_mae = []
                for deg in DIRECTIONS_DEG:
                    deg_str = str(float(deg))
                    for t in _get_trials(force_results, mag_str, deg_str, elev_str, is_v2):
                        if t["success"] and t.get("estimator_metrics"):
                            all_mae.append(t["estimator_metrics"]["mae"])
                elev_maes.append(np.mean(all_mae) if all_mae else 0.0)
            ax.plot(elevation_angles, elev_maes, marker="o", linewidth=1.5, label=f"{mag:.0f}N")
        ax.set_xlabel("Elevation Angle (deg)")
        ax.set_ylabel("MAE (N)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures.append(fig)

        # Peak displacement vs elevation angle
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"Peak Displacement vs Elevation Angle — {task_name}", fontsize=13, fontweight="bold")
        for mag in magnitudes:
            mag_str = str(float(mag))
            elev_disp = []
            for elev in elevation_angles:
                elev_str = str(float(elev))
                all_disp = []
                for deg in DIRECTIONS_DEG:
                    deg_str = str(float(deg))
                    for t in _get_trials(force_results, mag_str, deg_str, elev_str, is_v2):
                        if t["success"]:
                            all_disp.append(t["peak_displacement"])
                elev_disp.append(np.mean(all_disp) if all_disp else 0.0)
            ax.plot(elevation_angles, elev_disp, marker="o", linewidth=1.5, label=f"{mag:.0f}N")
        ax.set_xlabel("Elevation Angle (deg)")
        ax.set_ylabel("Peak Displacement (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures.append(fig)

        # Fz MAE vs elevation angle (for force_dim >= 3)
        if force_dim >= 3 and force_layout != "xy_yaw":
            fz_idx = 2
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(f"Fz MAE vs Elevation Angle — {task_name}", fontsize=13, fontweight="bold")
            for mag in magnitudes:
                mag_str = str(float(mag))
                fz_maes = []
                for elev in elevation_angles:
                    elev_str = str(float(elev))
                    all_fz = []
                    for deg in DIRECTIONS_DEG:
                        deg_str = str(float(deg))
                        for t in _get_trials(force_results, mag_str, deg_str, elev_str, is_v2):
                            if t["success"] and t.get("estimator_metrics"):
                                all_fz.append(t["estimator_metrics"]["per_axis_mae"][fz_idx])
                    fz_maes.append(np.mean(all_fz) if all_fz else 0.0)
                ax.plot(elevation_angles, fz_maes, marker="o", linewidth=1.5, label=f"{mag:.0f}N")
            ax.set_xlabel("Elevation Angle (deg)")
            ax.set_ylabel("Fz MAE (N)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            figures.append(fig)

    # ── Torque sweep pages ──────────────────────────────────────────
    if has_torque and has_estimator:
        torque_axes = list(torque_results.keys())
        torque_mags = sorted(set(float(m) for ax_data in torque_results.values() for m in ax_data.keys()))

        # Torque estimate vs GT time series (one page per axis)
        for axis_name in torque_axes:
            n_tmags = len(torque_mags)
            fig, axes_grid = plt.subplots(2, n_tmags, figsize=(5 * n_tmags, 8))
            if n_tmags == 1:
                axes_grid = axes_grid.reshape(2, 1)
            fig.suptitle(f"Torque Estimate vs GT — {axis_name} — {task_name}",
                         fontsize=14, fontweight="bold")
            for col, tmag in enumerate(torque_mags):
                tmag_str = str(float(tmag))
                for row, sign_str in enumerate(["+", "-"]):
                    ax = axes_grid[row, col]
                    trials = torque_results[axis_name].get(tmag_str, {}).get(sign_str, [])
                    succ_trials = [t for t in trials if t["success"] and "force_est" in t]
                    if succ_trials:
                        t0 = succ_trials[0]
                        est = np.array(t0["force_est"])
                        gt = _build_gt_vector(t0, force_dim, force_layout, is_torque=True)
                        time_s = np.arange(len(est)) * dt
                        colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown", "tab:orange"]
                        for d in range(force_dim):
                            ax.axhline(gt[d], color=colors[d % len(colors)], linestyle="--", alpha=0.4)
                            ax.plot(time_s, est[:, d], color=colors[d % len(colors)], linewidth=0.8,
                                    label=dim_labels[d] if row == 0 and col == 0 else None)
                    ax.set_title(f"{sign_str}{tmag:.0f} Nm", fontsize=10)
                    ax.set_xlabel("Time (s)", fontsize=8)
                    ax.set_ylabel("Est / GT", fontsize=8)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.3)
                    if row == 0 and col == 0:
                        ax.legend(fontsize=5, ncol=2)
            plt.tight_layout()
            figures.append(fig)

        # Torque MAE bar chart (grouped by axis and magnitude)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title(f"Torque Sweep — Per-Axis MAE — {task_name}", fontsize=13, fontweight="bold")
        bar_data = []
        bar_labels = []
        for axis_name in torque_axes:
            est_dim_idx = {"roll": 3, "pitch": 4, "yaw": 5}
            if force_layout == "xy_yaw":
                est_dim_idx = {"yaw": 2}
            elif force_dim == 4:
                est_dim_idx = {"yaw": 3}

            target_idx = est_dim_idx.get(axis_name)
            if target_idx is None or target_idx >= force_dim:
                continue

            for tmag in torque_mags:
                tmag_str = str(float(tmag))
                all_mae = []
                for sign_str in ["+", "-"]:
                    for t in torque_results[axis_name].get(tmag_str, {}).get(sign_str, []):
                        if t["success"] and t.get("estimator_metrics"):
                            all_mae.append(t["estimator_metrics"]["per_axis_mae"][target_idx])
                bar_data.append(np.mean(all_mae) if all_mae else 0.0)
                bar_labels.append(f"{axis_name}\n{tmag:.0f}Nm")

        if bar_data:
            x = np.arange(len(bar_data))
            ax.bar(x, bar_data, color="tab:purple", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, fontsize=9)
            ax.set_ylabel("MAE (Nm)")
            ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        figures.append(fig)

        # Torque bias bar chart (mean signed error)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title(f"Torque Sweep — Per-Axis Bias — {task_name}", fontsize=13, fontweight="bold")
        bias_data = []
        bias_labels = []
        for axis_name in torque_axes:
            est_dim_idx = {"roll": 3, "pitch": 4, "yaw": 5}
            if force_layout == "xy_yaw":
                est_dim_idx = {"yaw": 2}
            elif force_dim == 4:
                est_dim_idx = {"yaw": 3}

            target_idx = est_dim_idx.get(axis_name)
            if target_idx is None or target_idx >= force_dim:
                continue

            for tmag in torque_mags:
                tmag_str = str(float(tmag))
                for sign_str, sign_label in [("+", "+"), ("-", "-")]:
                    all_bias = []
                    for t in torque_results[axis_name].get(tmag_str, {}).get(sign_str, []):
                        if t["success"] and t.get("estimator_metrics"):
                            all_bias.append(t["estimator_metrics"]["per_axis_bias"][target_idx])
                    bias_data.append(np.mean(all_bias) if all_bias else 0.0)
                    bias_labels.append(f"{axis_name}\n{sign_label}{tmag:.0f}Nm")

        if bias_data:
            x = np.arange(len(bias_data))
            colors = ["tab:green" if b >= 0 else "tab:red" for b in bias_data]
            ax.bar(x, bias_data, color=colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(bias_labels, fontsize=8)
            ax.set_ylabel("Bias (Nm)")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        figures.append(fig)

    # ── Summary table ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 2 + len(magnitudes) * 0.6))
    ax.axis("off")
    headers = ["Mag (N)", "Success %", "Peak Disp (m)", "Mean Disp (m)"]
    if has_estimator:
        headers.extend(["MAE (N)", "Median AE (N)", "Rel Err %"])
        for dl in dim_labels:
            headers.append(f"MAE {dl}")

    rows = []
    for mag in magnitudes:
        mag_str = str(float(mag))
        all_trials = []
        for deg in DIRECTIONS_DEG:
            deg_str = str(float(deg))
            all_trials.extend(_get_trials(force_results, mag_str, deg_str, elev0_str, is_v2))
        succ = [t for t in all_trials if t["success"]]
        if succ:
            row = [
                f"{mag:.0f}",
                f"{np.mean([t['success'] for t in all_trials]) * 100:.0f}%",
                f"{np.mean([t['peak_displacement'] for t in succ]):.3f} ± {np.std([t['peak_displacement'] for t in succ]):.3f}",
                f"{np.mean([t['mean_displacement'] for t in succ]):.3f} ± {np.std([t['mean_displacement'] for t in succ]):.3f}",
            ]
            if has_estimator:
                m = [t["estimator_metrics"] for t in succ if t.get("estimator_metrics")]
                if m:
                    row.extend([
                        f"{np.mean([x['mae'] for x in m]):.2f}",
                        f"{np.mean([x['median_ae'] for x in m]):.2f}",
                        f"{np.mean([x['relative_err'] for x in m if 'relative_err' in x]):.1f}",
                    ])
                    for d_idx in range(force_dim):
                        row.append(f"{np.mean([x['per_axis_mae'][d_idx] for x in m]):.2f}")
                else:
                    row.extend(["N/A"] * (3 + force_dim))
            rows.append(row)
        else:
            row = [f"{mag:.0f}", "0%", "N/A", "N/A"]
            if has_estimator:
                row.extend(["N/A"] * (3 + force_dim))
            rows.append(row)

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    fig.suptitle(f"Static 360 Evaluation Summary (elev=0°) — {task_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    figures.append(fig)

    # ── Save PDF ────────────────────────────────────────────────────
    from matplotlib.backends.backend_pdf import PdfPages
    out_path = os.path.join(args.run_dir, "analysis.pdf")
    with PdfPages(out_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n  Analysis saved to {out_path}")
    print(f"  Pages: {len(figures)}")


if __name__ == "__main__":
    main()
