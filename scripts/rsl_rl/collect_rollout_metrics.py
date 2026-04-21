"""Collect rollout estimator metrics from all ablation runs into a single master JSON.

Scans data/eval/<experiment_folder>/ for the latest rollout_estimator_* subfolder,
reads metrics.json, and writes data/eval/master_metrics.json.

Usage:
    python scripts/rsl_rl/collect_rollout_metrics.py
    python scripts/rsl_rl/collect_rollout_metrics.py --eval_root data/eval --out data/eval/master_metrics.json
"""

import argparse
import json
import os
from pathlib import Path

# ── Run registry ─────────────────────────────────────────────────────────────
# (run_id, experiment_folder, description)
# description: training force range | estimated dims | special rewards / variants

RUNS = [
    # ── A-series: 3D XYZ, varying history length ───────────────────────────
    ("A1", "ablation_A1_h10_3d_rec",
     "3D (Fx,Fy,Fz) | h=10 | max_force=20N | rec loss w=1"),
    ("A2", "ablation_A2_h40_3d_rec",
     "3D (Fx,Fy,Fz) | h=40 | max_force=20N | rec loss w=1"),

    # ── B-series: 6D wrench ────────────────────────────────────────────────
    ("B2", "ablation_B2_h40_6d_rec",
     "6D wrench | h=40 | max_force=20N, max_torque=5Nm | rec loss w=1"),
    ("B3", "ablation_B3_h40_6d_rec_big",
     "6D wrench | h=40 | max_force=20N, max_torque=5Nm | rec loss w=1 | bigger net enc=[256,128]"),

    # ── J-series: 4D with estimation accuracy reward ───────────────────────
    ("J1", "ablation_J1_4d_h40_estrew_w1_30N",
     "4D (Fx,Fy,Fz,τ_yaw) | h=40 | max_force=30N, max_torque=10Nm | est accuracy reward w=1"),
    ("J3", "ablation_J3_4d_h40_estrew_w50_30N",
     "4D (Fx,Fy,Fz,τ_yaw) | h=40 | max_force=30N, max_torque=10Nm | est accuracy reward w=50"),
    ("J5", "ablation_J5_4d_h40_estrew_w50_tcnpre_30N",
     "4D (Fx,Fy,Fz,τ_yaw) | h=40 | max_force=30N, max_torque=10Nm | est accuracy reward w=50 | TCN preprocessor"),

    # ── P-series: PAINT trapezoid wrench, 30N / 10Nm (P19: 5Nm) ───────────
    ("P1",  "ablation_P1_h10_4d",
     "4D (Fx,Fy,Fz,τ_yaw) | h=10 | max_force=30N, max_torque=10Nm | PAINT wrench"),
    ("P2",  "ablation_P2_h20_4d",
     "4D (Fx,Fy,Fz,τ_yaw) | h=20 | max_force=30N, max_torque=10Nm | PAINT wrench"),
    ("P3",  "ablation_P3_h30_4d",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | PAINT wrench [baseline]"),
    ("P4",  "ablation_P4_h40_4d",
     "4D (Fx,Fy,Fz,τ_yaw) | h=40 | max_force=30N, max_torque=10Nm | PAINT wrench"),
    ("P5",  "ablation_P5_h30_4d_half",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | smaller net enc=[64,32]"),
    ("P6",  "ablation_P6_h30_4d_double",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | bigger net enc=[256,128]"),
    ("P7",  "ablation_P7_h30_4d_estrew_w50",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | est accuracy reward w=50"),
    ("P8",  "ablation_P8_h30_4d_comp_w0p5",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | compliance reward w=0.5"),
    ("P9",  "ablation_P9_h30_4d_comp_w1p0",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | compliance reward w=1.0"),
    ("P10", "ablation_P10_h30_4d_comp_w5p0",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | compliance reward w=5.0"),
    ("P11", "ablation_P11_h30_4d_norec",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | no reconstruction loss"),
    ("P12", "ablation_P12_h30_4d_tcn",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | TCN encoder"),
    ("P13", "ablation_P13_h30_2d",
     "2D (Fx,Fy) | h=30 | max_force=30N, max_torque=10Nm | 2D estimator"),
    ("P14", "ablation_P14_h30_xy_yaw",
     "3D xy_yaw (Fx,Fy,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | xy+yaw layout"),
    ("P15", "ablation_P15_h30_4d",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | dimension ablation ctrl"),
    ("P16", "ablation_P16_h30_6d",
     "6D wrench | h=30 | max_force=30N, max_torque=10Nm | torque angle loss w=3, yaw loss w=3"),
    ("P17", "ablation_P17_h30_6d_big",
     "6D wrench | h=30 | max_force=30N, max_torque=10Nm | bigger net enc=[256,128] | torque angle loss"),
    ("P19", "ablation_P19_h30_4d_torque5nm",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=5Nm | lower torque range"),
    ("P20", "ablation_P20_h30_4d_pd25",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | stiffer PD Kp=25, Kd=0.5"),

    # ── P23-P35: PAINT trapezoid wrench, 30N / 10Nm ───────────────────────
    ("P23", "ablation_P23_h4_4d_paint_trap",
     "4D (Fx,Fy,Fz,τ_yaw) | h=4 | max_force=30N, max_torque=10Nm | PAINT trap"),
    ("P24", "ablation_P24_h8_4d_paint_trap",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap [h=8 baseline]"),
    ("P25", "ablation_P25_h10_4d_paint_trap",
     "4D (Fx,Fy,Fz,τ_yaw) | h=10 | max_force=30N, max_torque=10Nm | PAINT trap"),
    ("P26", "ablation_P26_h8_4d_paint_trap_norec",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap | no rec loss"),
    ("P27", "ablation_P27_h8_4d_paint_trap_zero20",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap | 20% zero-force envs"),
    ("P28", "ablation_P28_h8_4d_constant",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | constant force profile"),
    ("P29", "ablation_P29_h8_4d_cyclic_trap",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | cyclic trap profile"),
    ("P30", "ablation_P30_h4_4d_constant",
     "4D (Fx,Fy,Fz,τ_yaw) | h=4 | max_force=30N, max_torque=10Nm | constant force profile"),
    ("P31", "ablation_P31_h8_4d_constant_b",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | constant force profile variant b"),
    ("P32a", "ablation_P32a_h8_4d_paint_trap_comp_b30_s025",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap | compliance B=30 σ=0.25"),
    ("P32b", "ablation_P32b_h8_4d_paint_trap_comp_b30_s05",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap | compliance B=30 σ=0.5"),
    ("P32c", "ablation_P32c_h8_4d_paint_trap_comp_b40_s025",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap | compliance B=40 σ=0.25"),
    ("P32d", "ablation_P32d_h8_4d_paint_trap_comp_b40_s05",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap | compliance B=40 σ=0.5"),
    ("P33", "ablation_P33_h8_4d_paint_trap_estacc",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap | est accuracy reward w=50"),
    ("P34", "ablation_P34_h8_4d_paint_trap_both_b30_s025",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap | compliance B=30 σ=0.25 + est acc w=50"),
    ("P35", "ablation_P35_h8_4d_paint_trap_estacc_tcn",
     "4D (Fx,Fy,Fz,τ_yaw) | h=8 | max_force=30N, max_torque=10Nm | PAINT trap | est acc w=50 + TCN pre"),

    # ── Q-series: bigger net + TCN + no rec + est_acc, PAINT trapezoid ────
    ("Q1", "ablation_Q1_h10_4d_paint_trap_big_tcn_norec_estacc",
     "4D (Fx,Fy,Fz,τ_yaw) | h=10 | max_force=30N, max_torque=10Nm | bigger net enc=[256,128] | TCN pre | no rec | est acc w=50"),
    ("Q2", "ablation_Q2_h30_4d_paint_trap_big_tcn_norec_estacc",
     "4D (Fx,Fy,Fz,τ_yaw) | h=30 | max_force=30N, max_torque=10Nm | bigger net enc=[256,128] | TCN pre | no rec | est acc w=50"),
]


def find_latest_rollout_eval(exp_dir: Path) -> Path | None:
    """Return path to latest rollout_estimator_* subfolder, or None."""
    candidates = sorted(exp_dir.glob("rollout_estimator_*"))
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", default="data/eval")
    parser.add_argument("--out", default="data/eval/master_metrics.json")
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    master = {}
    missing = []

    for run_id, exp_folder, description in RUNS:
        exp_dir = eval_root / exp_folder
        if not exp_dir.exists():
            print(f"[MISSING dir]  {run_id}: {exp_dir}")
            missing.append(run_id)
            continue

        rollout_dir = find_latest_rollout_eval(exp_dir)
        if rollout_dir is None:
            print(f"[NO ROLLOUT]   {run_id}: no rollout_estimator_* in {exp_dir}")
            missing.append(run_id)
            continue

        metrics_path = rollout_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"[NO METRICS]   {run_id}: {metrics_path}")
            missing.append(run_id)
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        master[run_id] = {
            "run_id": run_id,
            "experiment_folder": exp_folder,
            "rollout_eval_dir": rollout_dir.name,
            "description": description,
            "metrics": metrics,
        }
        basket = metrics.get("baskets", {}).get("training", {})
        mae = basket.get("mae", float("nan"))
        ang = basket.get("angular_err_xy_deg_median", float("nan"))
        print(f"[OK] {run_id:>4s}  MAE={mae:.3f}  AngErr={ang:.1f}°  ← {rollout_dir.name}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(master, f, indent=2)

    print(f"\nSaved {len(master)} runs → {out_path}")
    if missing:
        print(f"Missing / skipped: {missing}")


if __name__ == "__main__":
    main()
