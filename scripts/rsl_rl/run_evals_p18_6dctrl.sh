#!/bin/bash
# Full eval pipeline for P18 and 6Dctrl-Total50 variants.
# Runs static, OU force, and rollout estimator evals headless.
set -e
cd "$(dirname "$0")/../.."

BASE=/home/ubuntu/go2_rl_lab/logs/rsl_rl

P18=$BASE/ablation_P18_h30_6d_big_tcn_norec_payload/2026-04-25_23-28-49/model_15000.pt
C_T50=$BASE/ablation_6Dctrl_curr_total50/2026-04-26_18-23-51/model_9500.pt
C_W10=$BASE/ablation_6Dctrl_curr_total50_estacc_w10/2026-04-26_18-23-51/model_9500.pt
C_W25=$BASE/ablation_6Dctrl_curr_total50_estacc_w25/2026-04-26_18-23-51/model_9500.pt
C_W50=$BASE/ablation_6Dctrl_curr_total50_estacc_w50/2026-04-26_18-23-51/model_9500.pt

run_eval() {
    echo "=========================================="
    echo "Running: $1"
    echo "=========================================="
    eval "$1"
    echo ""
}

run_suite() {
    local task="$1"
    local ckpt="$2"
    echo ""
    echo "##########################################"
    echo "  $task"
    echo "##########################################"
    run_eval "python scripts/rsl_rl/static_eval.py --task $task --checkpoint $ckpt --num_envs 12 --show_est --force_min 10.0 --force_max 30.0 --headless"
    run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task $task --checkpoint $ckpt --num_envs 12 --show_est --show_gt --duration 20 --headless"
    run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task $task --checkpoint $ckpt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"
}

run_suite "Go2-Ablation-P18-v0"                     "$P18"
run_suite "Go2-Ablation-6Dctrl-Total50-v0"          "$C_T50"
run_suite "Go2-Ablation-6Dctrl-Total50-EstAccW10-v0" "$C_W10"
run_suite "Go2-Ablation-6Dctrl-Total50-EstAccW25-v0" "$C_W25"
run_suite "Go2-Ablation-6Dctrl-Total50-EstAccW50-v0" "$C_W50"

echo "=========================================="
echo "All P18 + 6Dctrl-Total50 evals complete."
echo "=========================================="

echo "Updating master metrics..."
python scripts/rsl_rl/collect_rollout_metrics.py

echo ""
echo "Launch dashboard with:"
echo "  python scripts/rsl_rl/dashboard.py"
