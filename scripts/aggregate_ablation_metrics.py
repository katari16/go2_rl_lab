"""Aggregate force-estimator ablation metrics into one JSON for the report.

Collects, per ablation run:
  - rollout_estimator_eval metrics  (training_regime, 4096 envs)
  - ou_force_eval metrics           (OU disturbance, 4096 envs, 20 s)
Both with per-axis MAE + std dev and angular error + std dev.

Groups runs by what ablation axis they study, with a description per group.

Usage:
    python scripts/aggregate_ablation_metrics.py [--out report_ablation_metrics.json]
"""

import argparse
import glob
import json
import os

ROOT = "/home/ubuntu/go2_rl_lab"
P_SERIES = f"{ROOT}/logs/rsl_rl/ablations_p_series"
J_SERIES = f"{ROOT}/logs/rsl_rl/ablation_force_accuracy_reward"
ROLLOUT_ROOT = f"{ROOT}/data/eval"

# ── Run registry: short_id -> (log_dir_glob, rollout_dir_name, est config notes) ──
RUNS = {
    # history sweep
    "P1":  dict(log=f"{P_SERIES}/ablation_P1_h10_4d/2026*",   rollout="ablation_P1_h10_4d",
                cfg="h=10, 4D, net [128,64]/[32,16], rec_w=1, const wrench, no est-acc reward"),
    "P2":  dict(log=f"{P_SERIES}/ablation_P2_h20_4d/2026*",   rollout="ablation_P2_h20_4d",
                cfg="h=20, 4D, net [128,64]/[32,16], rec_w=1, const wrench, no est-acc reward"),
    "P3":  dict(log=f"{P_SERIES}/ablation_P3_h30_4d/2026*",   rollout="ablation_P3_h30_4d",
                cfg="h=30, 4D, net [128,64]/[32,16], rec_w=1, const wrench, no est-acc reward (P-series baseline)"),
    "P4":  dict(log=f"{P_SERIES}/ablation_P4_h40_4d/2026*",   rollout="ablation_P4_h40_4d",
                cfg="h=40, 4D, net [128,64]/[32,16], rec_w=1, const wrench, no est-acc reward"),
    # network size (4D)
    "P5":  dict(log=f"{P_SERIES}/ablation_P5_h30_4d_half/2026*",   rollout="ablation_P5_h30_4d_half",
                cfg="h=30, 4D, net [64,32]/[16,8] (half), rec_w=1"),
    "P6":  dict(log=f"{P_SERIES}/ablation_P6_h30_4d_double/2026*", rollout="ablation_P6_h30_4d_double",
                cfg="h=30, 4D, net [256,128]/[64,32] (double), rec_w=1"),
    # rec loss (P series, no est-acc reward)
    "P11": dict(log=f"{P_SERIES}/ablation_P11_h30_4d_norec/2026*", rollout="ablation_P11_h30_4d_norec",
                cfg="h=30, 4D, net [128,64]/[32,16], rec_w=0 (no reconstruction loss)"),
    # force dim
    "P13": dict(log=f"{P_SERIES}/ablation_P13_h30_2d/2026*",     rollout="ablation_P13_h30_2d",
                cfg="h=30, 2D (Fx,Fy), net [128,64]/[32,16], rec_w=1"),
    "P14": dict(log=f"{P_SERIES}/ablation_P14_h30_xy_yaw/2026*", rollout="ablation_P14_h30_xy_yaw",
                cfg="h=30, 3D xy_yaw (Fx,Fy,τ_yaw), net [128,64]/[32,16], rec_w=1"),
    "P16": dict(log=f"{P_SERIES}/ablation_P16_h30_6d/2026*",     rollout="ablation_P16_h30_6d",
                cfg="h=30, 6D full wrench, net [128,64]/[32,16] (default), rec_w=1, +torque-angle loss"),
    "P17": dict(log=f"{P_SERIES}/ablation_P17_h30_6d_big/2026*", rollout="ablation_P17_h30_6d_big",
                cfg="h=30, 6D full wrench, net [256,128]/[64,32] (big), rec_w=1, +torque-angle loss"),
    # PD gains
    "P20": dict(log=f"{P_SERIES}/ablation_P20_h30_4d_pd25/2026*", rollout="ablation_P20_h30_4d_pd25",
                cfg="h=30, 4D, net [128,64]/[32,16], rec_w=1, PD gains Kp=25/Kd=0.5 (baseline uses Kp=8/Kd=0.4)"),
    # J series (with est-accuracy reward, h=40)
    "J3":  dict(log=f"{J_SERIES}/ablation_J3_4d_h40_estrew_w50_30N/2026*", rollout="ablation_J3_4d_h40_estrew_w50_30N",
                cfg="h=40, 4D, net [128,64]/[32,16], rec_w=1, est-acc reward w=50, no TCN (J-series baseline)"),
    "J5":  dict(log=f"{J_SERIES}/ablation_J5_4d_h40_estrew_w50_tcnpre_30N/2026*", rollout="ablation_J5_4d_h40_estrew_w50_tcnpre_30N",
                cfg="h=40, 4D, net [128,64]/[32,16], rec_w=1, est-acc reward w=50, TCN preprocessor [64,64] k=3 dil=[1,2]"),
    "J6":  dict(log=f"{J_SERIES}/ablation_J6_4d_h40_estrew_w50_norec_30N/2026*", rollout="ablation_J6_4d_h40_estrew_w50_norec_30N",
                cfg="h=40, 4D, net [128,64]/[32,16], rec_w=0 (no rec loss), est-acc reward w=50, no TCN"),
}

# ── Ablation groups: what each set of runs is meant to study ──────────────────
GROUPS = [
    dict(name="history_size",
         studies="Effect of proprioceptive observation-history length H on estimation accuracy. "
                 "Longer history gives more context to resolve the deflection response under a "
                 "persistent force, at the cost of input dimensionality.",
         baseline="P3",
         runs=["P1", "P2", "P3", "P4"],
         axis_values={"P1": "H=10", "P2": "H=20", "P3": "H=30", "P4": "H=40"}),
    dict(name="network_size_4d",
         studies="Effect of estimator network capacity (encoder + force-head widths) at 4D wrench. "
                 "Tests whether a larger MLP helps or whether the baseline is already saturated.",
         baseline="P3",
         runs=["P5", "P3", "P6"],
         axis_values={"P5": "half [64,32]/[16,8]", "P3": "baseline [128,64]/[32,16]", "P6": "double [256,128]/[64,32]"}),
    dict(name="network_size_6d",
         studies="Effect of network capacity at 6D full wrench — default vs. big net.",
         baseline="P16",
         runs=["P16", "P17"],
         axis_values={"P16": "default [128,64]/[32,16]", "P17": "big [256,128]/[64,32]"}),
    dict(name="reconstruction_loss",
         studies="Effect of the auxiliary next-observation reconstruction (prediction) loss. "
                 "Tests whether predicting future proprioception helps the encoder learn a "
                 "force-relevant representation. Two clean pairs: P-series (no est-acc reward) and "
                 "J-series (with est-acc reward).",
         baseline="P3",
         runs=["P3", "P11", "J3", "J6"],
         axis_values={"P3": "with rec (P)", "P11": "no rec (P)", "J3": "with rec (J)", "J6": "no rec (J)"}),
    dict(name="tcn",
         studies="Effect of adding a TCN (dilated causal conv) preprocessor before the encoder MLP, "
                 "to exploit the temporal structure of the force-induced response. Clean J3 vs J5 pair "
                 "(everything else identical: h40, 4D, est-acc reward w=50).",
         baseline="J3",
         runs=["J3", "J5"],
         axis_values={"J3": "no TCN", "J5": "TCN preprocessor"}),
    dict(name="force_dim",
         studies="Effect of the estimated wrench dimensionality: 2D (Fx,Fy) -> 3D xy_yaw (Fx,Fy,τ_yaw) "
                 "-> 4D (Fx,Fy,Fz,τ_yaw) -> 6D full wrench. Tests whether asking the estimator to "
                 "regress more components degrades the components we care about.",
         baseline="P3",
         runs=["P13", "P14", "P3", "P16", "P17"],
         axis_values={"P13": "2D", "P14": "3D xy_yaw", "P3": "4D", "P16": "6D default", "P17": "6D big"}),
    dict(name="pd_gains",
         studies="Effect of low PD gains (Kp=8/Kd=0.4) vs. conventional gains (Kp=25/Kd=0.5). "
                 "Low gains -> more joint deflection under external force -> richer proprioceptive "
                 "imprint for the estimator, plus wider exploration bandwidth during training. "
                 "Note the baseline (Kp=8) corresponds to P3 here.",
         baseline="P3",
         runs=["P3", "P20"],
         axis_values={"P3": "Kp=8 / Kd=0.4 (baseline)", "P20": "Kp=25 / Kd=0.5"}),
]


def _latest(glob_pat):
    hits = sorted(glob.glob(glob_pat))
    return hits[-1] if hits else None


def _latest_rollout_metrics(rollout_name):
    pat = f"{ROLLOUT_ROOT}/{rollout_name}/rollout_estimator_*/metrics.json"
    hits = sorted(glob.glob(pat))
    # prefer the most recent (the overnight re-run with std dev)
    return hits[-1] if hits else None


def _latest_ou_metrics(log_dir_glob):
    log_dir = _latest(log_dir_glob)
    if not log_dir:
        return None
    pat = f"{log_dir}/ou_eval/ou_metrics_*.json"
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


def _extract_rollout(path):
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path))
    # take the first (only) basket
    bk = list(d["baskets"].values())[0]
    return {
        "source_file": path,
        "force_dim": d.get("force_dim"),
        "n_envs": d.get("n_envs"),
        "duration_s": d.get("duration_s"),
        "mae": bk.get("mae"),
        "mae_std": bk.get("mae_std"),
        "median_ae": bk.get("median_ae"),
        "per_axis_mae": bk.get("per_axis_mae"),
        "per_axis_mae_std": bk.get("per_axis_mae_std"),
        "force_mae": bk.get("force_mae"),
        "force_mae_std": bk.get("force_mae_std"),
        "torque_mae": bk.get("torque_mae"),
        "torque_mae_std": bk.get("torque_mae_std"),
        "angular_err_xy_deg_mean": bk.get("angular_err_xy_deg_mean"),
        "angular_err_xy_deg_mean_std": bk.get("angular_err_xy_deg_mean_std"),
        "angular_err_xy_deg_median": bk.get("angular_err_xy_deg_median"),
        "angular_err_xy_deg_median_std": bk.get("angular_err_xy_deg_median_std"),
        "relative_err_pct": bk.get("relative_err_pct"),
        "force_range": bk.get("force_range"),
    }


def _extract_ou(path):
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path))
    m = d["metrics"]
    return {
        "source_file": path,
        "force_dim": d.get("force_dim"),
        "num_envs": d.get("num_envs"),
        "duration_s": d.get("duration_s"),
        "mae": m.get("mae_total"),
        "mae_std": m.get("mae_std"),
        "per_axis_mae": {k: v["mae"] for k, v in m.get("per_axis", {}).items()},
        "per_axis_mae_std": m.get("per_axis_mae_std"),
        "force_mae": m.get("force_mae"),
        "force_mae_std": m.get("force_mae_std"),
        "torque_mae": m.get("torque_mae"),
        "torque_mae_std": m.get("torque_mae_std"),
        "angular_err_xy_deg_mean": m.get("angular_err_xy_deg_mean"),
        "angular_err_xy_deg_mean_std": m.get("angular_err_xy_deg_mean_std"),
        "angular_err_xy_deg_median": m.get("angular_err_xy_deg_median"),
        "angular_err_xy_deg_median_std": m.get("angular_err_xy_deg_median_std"),
        "relative_err_pct": m.get("relative_err_pct"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/report_ablation_metrics.json")
    args = ap.parse_args()

    runs_out = {}
    for rid, info in RUNS.items():
        rollout = _extract_rollout(_latest_rollout_metrics(info["rollout"]))
        ou = _extract_ou(_latest_ou_metrics(info["log"]))
        runs_out[rid] = {
            "config": info["cfg"],
            "rollout_eval": rollout,
            "ou_eval": ou,
        }
        flag = []
        if rollout is None: flag.append("rollout MISSING")
        if ou is None: flag.append("OU MISSING")
        print(f"{rid:5s}  {'OK' if not flag else ' | '.join(flag)}")

    out = {
        "description": "Force-estimator ablation metrics for the report. Each run has "
                       "rollout_estimator_eval (training-regime force distribution, 4096 envs) and "
                       "ou_force_eval (Ornstein-Uhlenbeck disturbance, 4096 envs, 20 s) metrics. "
                       "All MAE values are per-dimension mean absolute error in N (forces) / Nm (torques); "
                       "std dev is computed as (per-env mean over time) -> std across envs. "
                       "Angular error is the XY-plane force-direction error in degrees.",
        "metric_glossary": {
            "mae": "overall per-dim MAE over all estimated components",
            "median_ae": "median of per-sample per-dim MAE (rollout only)",
            "per_axis_mae": "MAE per component (Fx, Fy, Fz, τ_yaw, ...)",
            "force_mae": "mean MAE over the force components only",
            "torque_mae": "mean MAE over the torque components only (null if none)",
            "angular_err_xy_deg_mean/median": "XY force-direction error, mean/median over time",
            "*_std": "standard deviation across the 4096 parallel environments of the per-env mean",
            "relative_err_pct": "||err|| / ||F_gt|| averaged where |F_gt|>1N",
        },
        "ablation_groups": GROUPS,
        "runs": runs_out,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
