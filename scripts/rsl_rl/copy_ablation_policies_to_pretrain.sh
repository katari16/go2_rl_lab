#!/bin/bash
# Copy exported ablation policies to deploy/pre_train/ for real-robot deployment.
set -e
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
DST="$ROOT/deploy/pre_train"

copy_run() {
    local name="$1"
    local src="$2"
    mkdir -p "$DST/$name"
    cp "$src/policy.pt"    "$DST/$name/policy.pt"
    cp "$src/estimator.pt" "$DST/$name/estimator.pt"
    echo "[OK] $name"
}

copy_run "ablation_j3"  "$ROOT/logs/rsl_rl/ablation_force_accuracy_reward/ablation_J3_4d_h40_estrew_w50_30N/2026-04-15_17-24-50/exported_policy"
copy_run "ablation_j5"  "$ROOT/logs/rsl_rl/ablation_force_accuracy_reward/ablation_J5_4d_h40_estrew_w50_tcnpre_30N/2026-04-15_17-24-57/exported_policy"
copy_run "ablation_p3"  "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P3_h30_4d/2026-04-19_11-11-09/exported_policy"
copy_run "ablation_p6"  "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P6_h30_4d_double/2026-04-19_11-11-32/exported_policy"
copy_run "ablation_p9"  "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P9_h30_4d_comp_w1p0/2026-04-19_11-11-36/exported_policy"
copy_run "ablation_p10" "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P10_h30_4d_comp_w5p0/2026-04-19_11-11-36/exported_policy"
copy_run "ablation_p17" "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P17_h30_6d_big/2026-04-19_11-11-07/exported_policy"
copy_run "payload_3kg"  "$ROOT/logs/rsl_rl/go2_lowlevel/2026-03-31_20-57-20/exported_policy"

# P-series — additions for real-world ablation evals
copy_run "ablation_p1"  "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P1_h10_4d/2026-04-19_11-13-01/exported_policy"
copy_run "ablation_p2"  "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P2_h20_4d/2026-04-19_11-12-59/exported_policy"
copy_run "ablation_p4"  "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P4_h40_4d/2026-04-19_11-11-09/exported_policy"
copy_run "ablation_p5"  "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P5_h30_4d_half/2026-04-19_11-12-50/exported_policy"
copy_run "ablation_p12" "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P12_h30_4d_tcn/2026-04-19_11-11-03/exported_policy"
copy_run "ablation_p20" "$ROOT/logs/rsl_rl/ablations_p_series/ablation_P20_h30_4d_pd25/2026-04-19_11-10-59/exported_policy"

# P18 — payload (6D wrench + R1 architecture)
copy_run "ablation_p18" "$ROOT/logs/rsl_rl/ablation_P18_h30_6d_big_tcn_norec_payload/2026-04-25_23-28-49/exported_policy"

# 6Dctrl curriculum variants (6D wrench, pose-command obs)
copy_run "ablation_6dctrl_total50"         "$ROOT/logs/rsl_rl/ablation_6Dctrl_curr_total50/2026-04-26_18-23-51/exported_policy"
copy_run "ablation_6dctrl_total50_estacc_w10" "$ROOT/logs/rsl_rl/ablation_6Dctrl_curr_total50_estacc_w10/2026-04-26_18-23-51/exported_policy"
copy_run "ablation_6dctrl_total50_estacc_w25" "$ROOT/logs/rsl_rl/ablation_6Dctrl_curr_total50_estacc_w25/2026-04-26_18-23-51/exported_policy"
copy_run "ablation_6dctrl_total50_estacc_w50" "$ROOT/logs/rsl_rl/ablation_6Dctrl_curr_total50_estacc_w50/2026-04-26_18-23-51/exported_policy"

echo ""
echo "All policies copied to $DST"
