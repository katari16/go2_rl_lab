#!/bin/bash
# Export all deploy-candidate policies via play_export.py.
# Payload policy is already exported — skipped here.
# R1/R2: add once cluster training completes.
set -e
cd "$(dirname "$0")/../.."

run_export() {
    echo "=========================================="
    echo "Exporting: $1"
    echo "=========================================="
    eval "$1"
}

# ── J-series ──────────────────────────────────────────────────────────────────
run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-J3-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_force_accuracy_reward/ablation_J3_4d_h40_estrew_w50_30N/2026-04-15_17-24-50/model_17000.pt"

run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-J5-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_force_accuracy_reward/ablation_J5_4d_h40_estrew_w50_tcnpre_30N/2026-04-15_17-24-57/model_16000.pt"

# ── P-series ──────────────────────────────────────────────────────────────────
run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P3-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P3_h30_4d/2026-04-19_11-11-09/model_11500.pt"

run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P6-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P6_h30_4d_double/2026-04-19_11-11-32/model_11500.pt"

run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P9-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P9_h30_4d_comp_w1p0/2026-04-19_11-11-36/model_11500.pt"

run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P10-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P10_h30_4d_comp_w5p0/2026-04-19_11-11-36/model_11500.pt"

run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-P17-v0 --num_envs 1 --headless --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P17_h30_6d_big/2026-04-19_11-11-07/model_12000.pt"

# ── R-series (uncomment once cluster training completes) ──────────────────────
# run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-R1-v0 --num_envs 1 --headless --checkpoint <R1_CHECKPOINT>"
# run_export "python scripts/rsl_rl/play_export.py --task Go2-Ablation-R2-v0 --num_envs 1 --headless --checkpoint <R2_CHECKPOINT>"

echo "=========================================="
echo "All exports complete."
echo "Payload policy already at:"
echo "  logs/rsl_rl/go2_lowlevel/2026-03-31_20-57-20/exported_policy"
echo "=========================================="
