#!/bin/bash
# Full eval pipeline for R1 and R2: static, OU force, rollout.
# Run headless. After completion, update master metrics and open dashboard.
set -e
cd "$(dirname "$0")/../.."

BASE=/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series
R1=$BASE/ablation_R1_h30_6d_big_tcn_norec/2026-04-21_19-15-56/model_9500.pt
R2=$BASE/ablation_R2_h30_6d_big_tcn_norec_estacc/2026-04-21_19-16-02/model_9500.pt

run_eval() {
    echo "=========================================="
    echo "Running: $1"
    echo "=========================================="
    eval "$1"
    echo ""
}

# ── Static eval ───────────────────────────────────────────────────────────────
run_eval "python scripts/rsl_rl/static_eval.py --task Go2-Ablation-R1-v0 --checkpoint $R1 --num_envs 12 --show_est --force_min 10.0 --force_max 30.0 --headless"
run_eval "python scripts/rsl_rl/static_eval.py --task Go2-Ablation-R2-v0 --checkpoint $R2 --num_envs 12 --show_est --force_min 10.0 --force_max 30.0 --headless"

# ── OU force eval ─────────────────────────────────────────────────────────────
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-R1-v0 --checkpoint $R1 --num_envs 12 --show_est --show_gt --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-R2-v0 --checkpoint $R2 --num_envs 12 --show_est --show_gt --duration 20 --headless"

# ── Rollout estimator eval ────────────────────────────────────────────────────
run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-R1-v0 --checkpoint $R1 --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"
run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-R2-v0 --checkpoint $R2 --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

echo "=========================================="
echo "All R-series evals complete."
echo "=========================================="

# ── Update master metrics ─────────────────────────────────────────────────────
echo "Updating master metrics..."
python scripts/rsl_rl/collect_rollout_metrics.py

echo ""
echo "Launch dashboard with:"
echo "  python scripts/rsl_rl/dashboard.py"
