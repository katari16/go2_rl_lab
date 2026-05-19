#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

run_eval() {
    echo "=========================================="
    echo "Running: $1"
    echo "=========================================="
    eval "$1"
    echo ""
}

# ── TCN ablation set ──────────────────────────────────────────────────────────
# P28: constant profile, h=8, no est-acc reward, no TCN (baseline)
run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-P28-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_P28_h8_4d_constant/2026-04-19_15-07-20/model_2.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

# P33: PAINT trap, h=8, est-acc reward w=50, no TCN
run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-P33-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_P33_h8_4d_paint_trap_estacc/2026-04-19_15-11-13/model_2.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

# P35: PAINT trap, h=8, est-acc reward w=50 + TCN preprocessor
run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-P35-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_P35_h8_4d_paint_trap_estacc_tcn/2026-04-19_15-30-21/model_2.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

# Q1: PAINT trap, h=10, bigger net, TCN pre, no rec loss, est-acc w=50
run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-Q1-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_Q1_h10_4d_paint_trap_big_tcn_norec_estacc/2026-04-20_00-19-03/model_2.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

# Q2: PAINT trap, h=30, bigger net, TCN pre, no rec loss, est-acc w=50
run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-Q2-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_Q2_h30_4d_paint_trap_big_tcn_norec_estacc/2026-04-20_00-19-40/model_2.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

echo "=========================================="
echo "All P28/P33/P35/Q1/Q2 rollout evals complete."
echo "=========================================="
