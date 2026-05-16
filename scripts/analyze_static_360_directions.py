"""Standalone directional-bias analysis for static_360 raw_data.json.

For each azimuth direction, aggregates trials whose force magnitude is in a chosen
range and reports (mean, std over trials) of: per-axis MAE, force MAE, torque MAE,
angular error (mean & median). Writes direction_metrics.json and a polar plot.

Also can aggregate several runs into one comparison JSON + overlaid polar plots.

Usage:
    # single run
    python scripts/analyze_static_360_directions.py <static_360_dir_or_raw_data.json> \
        [--mag_range 15 25] [--elevation 0]

    # several runs at once -> data/static_360_ablations/{direction_metrics_all.json, polar_*.png}
    python scripts/analyze_static_360_directions.py --collect P3 J5 P16 ...
"""

import argparse
import glob
import json
import math
import os

import numpy as np

ROOT = "/home/ubuntu/go2_rl_lab"
EVAL_ROOT = f"{ROOT}/data/eval"
OUT_DIR = f"{ROOT}/data/static_360_ablations"

# run_id -> the experiment subdir under data/eval that holds its static_360_* folders
RUN_EVAL_SUBDIR = {
    "P1": "ablation_P1_h10_4d", "P2": "ablation_P2_h20_4d", "P3": "ablation_P3_h30_4d",
    "P4": "ablation_P4_h40_4d", "P5": "ablation_P5_h30_4d_half", "P6": "ablation_P6_h30_4d_double",
    "P11": "ablation_P11_h30_4d_norec", "P13": "ablation_P13_h30_2d", "P14": "ablation_P14_h30_xy_yaw",
    "P16": "ablation_P16_h30_6d", "P17": "ablation_P17_h30_6d_big", "P20": "ablation_P20_h30_4d_pd25",
    "J3": "ablation_J3_4d_h40_estrew_w50_30N", "J5": "ablation_J5_4d_h40_estrew_w50_tcnpre_30N",
    "J6": "ablation_J6_4d_h40_estrew_w50_norec_30N",
}


def _build_dim_labels(force_dim, force_layout):
    if force_layout == "xy_yaw" and force_dim == 3:
        return ["Fx", "Fy", "τ_yaw"]
    if force_dim == 2: return ["Fx", "Fy"]
    if force_dim == 3: return ["Fx", "Fy", "Fz"]
    if force_dim == 4: return ["Fx", "Fy", "Fz", "τ_yaw"]
    if force_dim == 6: return ["Fx", "Fy", "Fz", "τ_roll", "τ_pitch", "τ_yaw"]
    return [f"d{i}" for i in range(force_dim)]


def compute_direction_metrics(force_sweep, force_dim, force_layout,
                              mag_min=15.0, mag_max=25.0, elevation=0.0):
    dim_labels = _build_dim_labels(force_dim, force_layout)
    elev_str = str(float(elevation))

    per_dir = {}
    for mag_str, dir_dict in force_sweep.items():
        try:
            mag = float(mag_str)
        except (ValueError, TypeError):
            continue
        if not (mag_min <= mag <= mag_max):
            continue
        for deg_str, elev_dict in dir_dict.items():
            # elev_dict may be {elev_str: [trials]} (v2) or directly a list (v1)
            if isinstance(elev_dict, dict):
                trials = elev_dict.get(elev_str, [])
            else:
                trials = elev_dict
            if trials:
                per_dir.setdefault(deg_str, []).extend(trials)

    directions = {}
    for deg_str, trials in per_dir.items():
        # per-trial aggregates (between-trial spread) ...
        pm, pax, pang_m, pang_md = [], {l: [] for l in dim_labels}, [], []
        # ... and per-trial WITHIN-trial noise (temporal std of est around GT)
        pnoise, pnoise_ax = [], {l: [] for l in dim_labels}
        pbias_ax = {l: [] for l in dim_labels}
        for tr in trials:
            if not tr.get("success", True) or "force_est" not in tr:
                continue
            est = np.asarray(tr["force_est"], dtype=np.float32)
            if est.ndim != 2 or est.shape[0] < 5 or est.shape[1] < force_dim:
                continue
            est = est[:, :force_dim]
            fxyz = tr.get("force_xyz") or (tr.get("force_xy", [0, 0]) + [0.0])
            fx, fy, fz = (fxyz + [0.0, 0.0, 0.0])[:3]
            gt = np.zeros(force_dim, np.float32); gt[0] = fx; gt[1] = fy
            if force_layout != "xy_yaw" and force_dim >= 3:
                gt[2] = fz
            err = est - gt[None, :]
            ae = np.abs(err)
            pm.append(float(ae.mean()))
            # within-trial noise: std of the est over time (around its own mean) — pure jitter
            pnoise.append(float(np.linalg.norm(est - est.mean(axis=0, keepdims=True), axis=1).std()))
            for d, l in enumerate(dim_labels):
                pax[l].append(float(ae[:, d].mean()))
                pnoise_ax[l].append(float(est[:, d].std()))
                pbias_ax[l].append(float(err[:, d].mean()))
            if math.hypot(fx, fy) > 1.0:
                ga = math.atan2(fy, fx); ea = np.arctan2(est[:, 1], est[:, 0])
                dd = np.arctan2(np.sin(ea - ga), np.cos(ea - ga))
                deg = np.abs(dd) * 180 / np.pi
                pang_m.append(float(deg.mean())); pang_md.append(float(np.median(deg)))
        if not pm:
            continue
        pa  = {l: float(np.mean(pax[l])) for l in dim_labels}
        pas = {l: float(np.std(pax[l]))  for l in dim_labels}                 # between-trial spread of the MAE
        p_noise_ax  = {l: float(np.mean(pnoise_ax[l])) for l in dim_labels}    # within-trial jitter (per axis)
        p_bias_ax   = {l: float(np.mean(pbias_ax[l]))  for l in dim_labels}    # signed bias (per axis)
        f_idx = [d for d, l in enumerate(dim_labels) if "τ" not in l]
        t_idx = [d for d, l in enumerate(dim_labels) if "τ" in l]
        directions[deg_str] = {
            "mae": float(np.mean(pm)), "mae_std": float(np.std(pm)),
            "per_axis_mae": pa, "per_axis_mae_std": pas,
            "per_axis_noise_std": p_noise_ax,        # within-trial temporal std of the estimate, per axis
            "per_axis_bias": p_bias_ax,              # mean signed error, per axis
            "noise_std": float(np.mean(pnoise)),     # within-trial jitter of the (multi-dim) estimate, magnitude
            "force_noise_std": float(np.mean([p_noise_ax[dim_labels[d]] for d in f_idx])) if f_idx else 0.0,
            "force_mae": float(np.mean([pa[dim_labels[d]] for d in f_idx])) if f_idx else 0.0,
            "force_mae_std": float(np.mean([pas[dim_labels[d]] for d in f_idx])) if f_idx else 0.0,
            "torque_mae": float(np.mean([pa[dim_labels[d]] for d in t_idx])) if t_idx else None,
            "torque_mae_std": float(np.mean([pas[dim_labels[d]] for d in t_idx])) if t_idx else None,
            "angular_err_xy_deg_mean": float(np.mean(pang_m)) if pang_m else 0.0,
            "angular_err_xy_deg_mean_std": float(np.std(pang_m)) if pang_m else 0.0,
            "angular_err_xy_deg_median": float(np.mean(pang_md)) if pang_md else 0.0,
            "angular_err_xy_deg_median_std": float(np.std(pang_md)) if pang_md else 0.0,
            "n_trials": len(pm),
        }
    return {"dim_labels": dim_labels, "mag_range": [mag_min, mag_max],
            "elevation": elevation, "directions": directions,
            "metric_notes": {
                "per_axis_mae_std": "between-trial spread of the per-axis MAE (how reproducible the directional bias is)",
                "per_axis_noise_std": "within-trial temporal std of the estimate around its own mean, per axis (the jitter you SEE)",
                "noise_std": "within-trial jitter of the full estimate vector (magnitude)",
                "per_axis_bias": "mean signed error per axis (est - gt), averaged over trials",
            }}


def _load_raw(path):
    if os.path.isdir(path):
        # newest static_360_* subdir's raw_data.json
        hits = sorted(glob.glob(f"{path}/static_360_*/raw_data.json")) or \
               sorted(glob.glob(f"{path}/raw_data.json"))
        if not hits:
            raise FileNotFoundError(f"No raw_data.json under {path}")
        path = hits[-1]
    d = json.load(open(path))
    if "force_sweep" in d:  # v2
        meta = d.get("metadata", {})
        return d["force_sweep"], meta.get("force_dim"), meta.get("force_layout", "auto"), path
    # v1: results may be under 'results' or be the top dict; force_dim inferred from a trial
    fs = d.get("results", d)
    # infer force_dim from first trial's force_est width
    fd = 4
    try:
        m0 = next(iter(fs.values())); d0 = next(iter(m0.values()))
        sub = d0 if isinstance(d0, list) else next(iter(d0.values()))
        fd = len(sub[0]["force_est"][0])
    except Exception:
        pass
    return fs, fd, "auto", path


def _polar_compare(run_metrics, metric_path, title, out_png):
    """run_metrics: dict run_id -> direction_metrics dict. metric_path like 'force_mae' or
    'per_axis_mae:Fx' or 'angular_err_xy_deg_median'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")
    colors = ["#1d3557", "#e76f51", "#2a9d8f", "#9b5de5", "#f4a261", "#457b9d", "#264653", "#84a98c"]
    for i, (rid, dm) in enumerate(run_metrics.items()):
        dirs = dm["directions"]
        degs = sorted(float(k) for k in dirs)
        if not degs:
            continue
        vals = []
        for dg in degs:
            entry = dirs[str(dg)]
            if ":" in metric_path:
                base, axis = metric_path.split(":", 1)
                vals.append((entry.get(base) or {}).get(axis, np.nan))
            else:
                vals.append(entry.get(metric_path, np.nan))
        # close the loop
        th = np.deg2rad(degs + [degs[0]])
        rr = vals + [vals[0]]
        ax.plot(th, rr, "-o", color=colors[i % len(colors)], label=rid, linewidth=1.8, markersize=4)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title(title, pad=20, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  polar: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="static_360 dir or raw_data.json (single-run mode)")
    ap.add_argument("--collect", nargs="+", help="run ids to collect+compare")
    ap.add_argument("--mag_range", type=float, nargs=2, default=[15.0, 25.0])
    ap.add_argument("--elevation", type=float, default=0.0)
    args = ap.parse_args()
    mn, mx = args.mag_range

    if args.collect:
        os.makedirs(OUT_DIR, exist_ok=True)
        all_metrics = {}
        for rid in args.collect:
            sub = RUN_EVAL_SUBDIR.get(rid, rid)
            cand = f"{EVAL_ROOT}/{sub}"
            try:
                fs, fd, fl, path = _load_raw(cand)
            except FileNotFoundError:
                print(f"{rid}: no static_360 raw_data found under {cand} — skip")
                continue
            dm = compute_direction_metrics(fs, fd, fl, mn, mx, args.elevation)
            dm["source_file"] = path
            all_metrics[rid] = dm
            print(f"{rid}: {len(dm['directions'])} directions  (force_dim={fd}, layout={fl})  from {os.path.basename(os.path.dirname(path))}")
        out_json = f"{OUT_DIR}/direction_metrics_all.json"
        with open(out_json, "w") as f:
            json.dump({"mag_range": [mn, mx], "elevation": args.elevation, "runs": all_metrics}, f, indent=2)
        print(f"Saved: {out_json}")
        if all_metrics:
            _polar_compare(all_metrics, "force_mae",
                           f"Force MAE vs direction  ([{mn:.0f},{mx:.0f}] N)", f"{OUT_DIR}/polar_force_mae.png")
            _polar_compare(all_metrics, "per_axis_mae:Fx",
                           f"MAE Fx vs direction  ([{mn:.0f},{mx:.0f}] N)", f"{OUT_DIR}/polar_mae_fx.png")
            _polar_compare(all_metrics, "per_axis_mae:Fy",
                           f"MAE Fy vs direction  ([{mn:.0f},{mx:.0f}] N)", f"{OUT_DIR}/polar_mae_fy.png")
            _polar_compare(all_metrics, "angular_err_xy_deg_median",
                           f"Angular err (median, deg) vs direction  ([{mn:.0f},{mx:.0f}] N)", f"{OUT_DIR}/polar_ang_median.png")
        return

    if not args.path:
        ap.error("provide a path or --collect")
    fs, fd, fl, path = _load_raw(args.path)
    dm = compute_direction_metrics(fs, fd, fl, mn, mx, args.elevation)
    dm["source_file"] = path
    out_dir = os.path.dirname(path)
    out_json = os.path.join(out_dir, "direction_metrics.json")
    with open(out_json, "w") as f:
        json.dump(dm, f, indent=2)
    print(f"Saved: {out_json}  ({len(dm['directions'])} directions, force_dim={fd})")
    label = list(dm["dim_labels"])
    plots = [("force_mae", f"Force MAE vs direction ([{mn:.0f},{mx:.0f}] N)", "polar_force_mae.png"),
             ("per_axis_mae:Fx", f"MAE Fx vs direction ([{mn:.0f},{mx:.0f}] N)", "polar_mae_fx.png"),
             ("per_axis_mae:Fy", f"MAE Fy vs direction ([{mn:.0f},{mx:.0f}] N)", "polar_mae_fy.png"),
             ("angular_err_xy_deg_median", f"Angular err (median, deg) vs direction ([{mn:.0f},{mx:.0f}] N)", "polar_ang_median.png"),
             ("mae", f"Overall per-dim MAE vs direction ([{mn:.0f},{mx:.0f}] N)", "polar_mae.png")]
    if "Fz" in label:
        plots.append(("per_axis_mae:Fz", f"MAE Fz vs direction ([{mn:.0f},{mx:.0f}] N)", "polar_mae_fz.png"))
    if "τ_yaw" in label:
        plots.append(("per_axis_mae:τ_yaw", f"MAE τ_yaw vs direction ([{mn:.0f},{mx:.0f}] N)", "polar_mae_tauyaw.png"))
    for mp, title, fn in plots:
        _polar_compare({"R1": dm}, mp, title, os.path.join(out_dir, fn))


if __name__ == "__main__":
    main()
