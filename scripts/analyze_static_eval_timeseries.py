"""Analyze static_eval timeseries (env-0 trajectories) for the ablation report.

Reads data/static_eval_ablations/<run>/static_eval_data_*.json and computes,
per estimated channel (Fx, Fy, Fz, τ_yaw, ...):

  Accuracy / bias
    mae                 mean |est - gt|
    bias                mean (est - gt)            (signed offset)
    rmse                sqrt(mean (est - gt)^2)

  Steady-state noise   (settled portion: skip SETTLE_S after each force change)
    ss_mae              MAE over settled samples
    ss_noise_std        std of (est - gt) over settled samples       (jitter incl. slow drift)
    step_noise_std      std of diff(est) over settled samples / sqrt2 (high-freq jitter)

  Transient tracking   (first SETTLE_S after each force change)
    transient_mae       MAE over transient samples
    settling_time_s     mean over transitions: time for |est - gt_new| to fall and
                        stay below SETTLE_FRAC * |Δgt|  (capped at segment length)
    overshoot_pct       mean over transitions: max excursion of est past gt_new,
                        as % of |Δgt|  (only counted where |Δgt| > MIN_STEP)

  Integral / area
    integral_abs_err    ∫|est - gt| dt          (N·s — area between the curves)
    integral_signed_err ∫(est - gt) dt          (N·s — net signed area = bias·T)
    rel_integral_err    ∫|est - gt| dt / ∫|gt| dt   (dimensionless)

Aggregate (across channels): per-dim mae, force_mae, torque_mae,
angular_err_xy_deg_{mean,median}, and mean over force channels of
ss_noise_std / settling_time_s / rel_integral_err.

Output: data/static_eval_ablations/static_eval_timeseries_metrics.json
        (same group structure as report_ablation_metrics.json).
"""

import glob
import json
import math
import os

import numpy as np

ROOT = "/home/ubuntu/go2_rl_lab"
DEST = f"{ROOT}/data/static_eval_ablations"

SETTLE_S    = 0.5    # seconds after a force change considered "transient"
SETTLE_FRAC = 0.20   # |est - gt_new| must fall below this fraction of |Δgt| to be "settled"
MIN_STEP    = 2.0    # only analyse transitions whose GT jump magnitude exceeds this (N or Nm)

# (channel label, gt key, est key)
CHANNEL_SPECS = [
    ("Fx",     "gt_force_x",     "est_force_x"),
    ("Fy",     "gt_force_y",     "est_force_y"),
    ("Fz",     "gt_force_z",     "est_force_z"),
    ("τ_roll", "gt_torque_roll", "est_torque_roll"),
    ("τ_pitch","gt_torque_pitch","est_torque_pitch"),
    ("τ_yaw",  "gt_torque_yaw",  "est_torque_yaw"),
]

# import the ablation groups + run registry from the sim-eval aggregator so the
# structure stays consistent
import importlib.util
_spec = importlib.util.spec_from_file_location("agg", f"{ROOT}/scripts/aggregate_ablation_metrics.py")
_agg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_agg)
GROUPS = _agg.GROUPS
RUN_IDS = list(_agg.RUNS.keys())


def _load_timeseries(run_id):
    hits = sorted(glob.glob(f"{DEST}/{run_id}/static_eval_data_*.json"))
    if not hits:
        return None, None
    path = hits[-1]
    return json.load(open(path)), path


def _segments(n_steps, rerandom_steps):
    """Return list of (start, end) segment index ranges between force changes."""
    cuts = sorted(int(s) for s in rerandom_steps if 0 < int(s) < n_steps)
    bounds = [0] + cuts + [n_steps]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


def _channel_metrics(gt, est, dt, segments, settle_n):
    err = est - gt
    abs_err = np.abs(err)

    mae  = float(abs_err.mean())
    bias = float(err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))

    # ── settled vs transient masks ──────────────────────────────────────
    settled_mask  = np.zeros(len(gt), dtype=bool)
    transient_mask = np.zeros(len(gt), dtype=bool)
    for (s, e) in segments:
        t_end = min(s + settle_n, e)
        transient_mask[s:t_end] = True
        if e - s > settle_n:
            settled_mask[s + settle_n:e] = True

    if settled_mask.any():
        ss_err = err[settled_mask]
        ss_mae = float(np.abs(ss_err).mean())
        ss_noise_std = float(ss_err.std())
    else:
        ss_mae = mae
        ss_noise_std = float(err.std())

    # step-to-step (high-freq) noise on the settled portion
    if settled_mask.sum() > 2:
        de = np.diff(est[settled_mask])
        step_noise_std = float(de.std() / math.sqrt(2.0))
    else:
        step_noise_std = float(np.diff(est).std() / math.sqrt(2.0))

    transient_mae = float(abs_err[transient_mask].mean()) if transient_mask.any() else mae

    # ── settling time + overshoot per transition ────────────────────────
    settling_times = []
    overshoots = []
    for k, (s, e) in enumerate(segments):
        if e - s < 3:
            continue
        gt_new = float(np.median(gt[s:e]))
        gt_old = float(np.median(gt[segments[k - 1][0]:segments[k - 1][1]])) if k > 0 else 0.0
        step = abs(gt_new - gt_old)
        if step < MIN_STEP:
            continue
        thr = SETTLE_FRAC * step
        seg_err = np.abs(est[s:e] - gt_new)
        # first index from which it stays below thr to the segment end
        settled_at = None
        below = seg_err <= thr
        for i in range(len(below)):
            if below[i:].all():
                settled_at = i
                break
        if settled_at is None:
            settling_times.append((e - s) * dt)  # never settled within the segment
        else:
            settling_times.append(settled_at * dt)
        # overshoot: how far est goes past gt_new, in the direction of the jump
        direction = math.copysign(1.0, gt_new - gt_old)
        excursion = (est[s:e] - gt_new) * direction
        max_over = float(excursion.max())
        overshoots.append(max(0.0, max_over) / step * 100.0)

    settling_time_s = float(np.mean(settling_times)) if settling_times else 0.0
    overshoot_pct   = float(np.mean(overshoots)) if overshoots else 0.0

    # ── integral / area metrics ─────────────────────────────────────────
    integral_abs_err    = float(abs_err.sum() * dt)
    integral_signed_err = float(err.sum() * dt)
    denom = float(np.abs(gt).sum())
    rel_integral_err    = float(abs_err.sum() / denom) if denom > 1e-6 else 0.0

    return {
        "mae": mae, "bias": bias, "rmse": rmse,
        "ss_mae": ss_mae, "ss_noise_std": ss_noise_std, "step_noise_std": step_noise_std,
        "transient_mae": transient_mae,
        "settling_time_s": settling_time_s, "overshoot_pct": overshoot_pct,
        "integral_abs_err": integral_abs_err, "integral_signed_err": integral_signed_err,
        "rel_integral_err": rel_integral_err,
        "n_transitions_analysed": len(settling_times),
    }


def _angular_metrics(gt_fx, gt_fy, est_fx, est_fy):
    xy_mag = np.hypot(gt_fx, gt_fy)
    mask = xy_mag > 1.0
    if not mask.any():
        return 0.0, 0.0
    gt_ang  = np.arctan2(gt_fy[mask], gt_fx[mask])
    est_ang = np.arctan2(est_fy[mask], est_fx[mask])
    diff = np.arctan2(np.sin(est_ang - gt_ang), np.cos(est_ang - gt_ang))
    deg = np.abs(diff) * 180.0 / np.pi
    return float(deg.mean()), float(np.median(deg))


def analyze_run(run_id):
    d, path = _load_timeseries(run_id)
    if d is None:
        return None
    t = np.asarray(d["time_s"], dtype=np.float64)
    n = len(t)
    dt = float(np.median(np.diff(t))) if n > 1 else 0.02
    settle_n = max(1, int(round(SETTLE_S / dt)))
    segments = _segments(n, d.get("rerandom_steps", []))

    channels = {}
    available = []
    for label, gk, ek in CHANNEL_SPECS:
        if gk in d and ek in d:
            gt = np.asarray(d[gk], dtype=np.float64)
            est = np.asarray(d[ek], dtype=np.float64)
            channels[label] = _channel_metrics(gt, est, dt, segments, settle_n)
            available.append(label)

    # aggregates
    force_labels  = [l for l in available if "τ" not in l]
    torque_labels = [l for l in available if "τ" in l]
    mae_all    = float(np.mean([channels[l]["mae"] for l in available])) if available else 0.0
    force_mae  = float(np.mean([channels[l]["mae"] for l in force_labels])) if force_labels else 0.0
    torque_mae = float(np.mean([channels[l]["mae"] for l in torque_labels])) if torque_labels else None
    ss_noise_force  = float(np.mean([channels[l]["ss_noise_std"] for l in force_labels])) if force_labels else 0.0
    settle_force    = float(np.mean([channels[l]["settling_time_s"] for l in force_labels])) if force_labels else 0.0
    rel_int_force   = float(np.mean([channels[l]["rel_integral_err"] for l in force_labels])) if force_labels else 0.0

    ang_mean = ang_med = 0.0
    if "Fx" in d if False else ("gt_force_x" in d and "gt_force_y" in d):
        ang_mean, ang_med = _angular_metrics(
            np.asarray(d["gt_force_x"]), np.asarray(d["gt_force_y"]),
            np.asarray(d["est_force_x"]), np.asarray(d["est_force_y"]))

    return {
        "source_file": path,
        "n_steps": n, "dt": dt, "n_force_changes": len(segments),
        "channels": channels,
        "aggregate": {
            "mae": mae_all,
            "force_mae": force_mae,
            "torque_mae": torque_mae,
            "angular_err_xy_deg_mean": ang_mean,
            "angular_err_xy_deg_median": ang_med,
            "mean_ss_noise_std_force": ss_noise_force,
            "mean_settling_time_s_force": settle_force,
            "mean_rel_integral_err_force": rel_int_force,
        },
    }


def main():
    runs = {}
    for rid in RUN_IDS:
        r = analyze_run(rid)
        runs[rid] = r
        print(f"{rid:5s}  {'OK' if r else 'MISSING'}"
              + (f"  agg_mae={r['aggregate']['mae']:.2f}  ss_noise(F)={r['aggregate']['mean_ss_noise_std_force']:.2f}"
                 f"  settle(F)={r['aggregate']['mean_settling_time_s_force']:.2f}s" if r else ""))

    out = {
        "description": "Static-eval (env-0, standing robot, persistent force re-randomised every 1-3 s "
                       f"over ~20 s, |F|∈[10,30] N/axis) timeseries analysis for the ablation report. "
                       "Metrics characterise estimator accuracy, steady-state noise, transient tracking "
                       "of force changes, and area between the est/GT curves. This is a single-trajectory "
                       "analysis (env 0), complementary to the population statistics in report_ablation_metrics.json.",
        "params": {"settle_s": SETTLE_S, "settle_frac": SETTLE_FRAC, "min_step_for_transition": MIN_STEP},
        "metric_glossary": {
            "mae/bias/rmse": "accuracy and signed offset over the whole trajectory",
            "ss_mae": "MAE on the settled portion (≥0.5 s after each force change)",
            "ss_noise_std": "std of (est-gt) on the settled portion — overall jitter incl. slow drift",
            "step_noise_std": "std of step-to-step change in est on the settled portion / √2 — high-freq jitter",
            "transient_mae": "MAE in the first 0.5 s after each force change",
            "settling_time_s": "mean time for |est-gt_new| to fall and stay below 20% of the GT jump",
            "overshoot_pct": "mean overshoot past the new GT value, % of the GT jump",
            "integral_abs_err": "∫|est-gt| dt — area between curves (N·s)",
            "integral_signed_err": "∫(est-gt) dt — net signed area = bias·T (N·s)",
            "rel_integral_err": "∫|est-gt| dt / ∫|gt| dt — dimensionless tracking error",
        },
        "ablation_groups": GROUPS,
        "runs": runs,
    }
    out_path = f"{DEST}/static_eval_timeseries_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
