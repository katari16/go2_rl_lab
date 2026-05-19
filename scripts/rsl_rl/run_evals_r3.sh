#!/bin/bash
# Full eval pipeline for R3 (R1 + fixed base mass): static, OU force, rollout.
# Run headless. After completion, update master metrics and open dashboard.
set -e
cd "$(dirname "$0")/../.."

BASE=/home/ubuntu/go2_rl_lab/logs/rsl_rl
R3=$BASE/ablation_R3_h30_6d_big_tcn_norec_nomassrand/2026-04-23_23-09-49/model_10000.pt

run_eval() {
    echo "=========================================="
    echo "Running: $1"
    echo "=========================================="
    eval "$1"
    echo ""
}

# ── Static eval ───────────────────────────────────────────────────────────────
run_eval "python scripts/rsl_rl/static_eval.py --task Go2-Ablation-R3-v0 --checkpoint $R3 --num_envs 12 --show_est --force_min 10.0 --force_max 30.0 --headless"

# ── OU force eval ─────────────────────────────────────────────────────────────
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-R3-v0 --checkpoint $R3 --num_envs 12 --show_est --show_gt --duration 20 --headless"

# ── Rollout estimator eval ────────────────────────────────────────────────────
run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-R3-v0 --checkpoint $R3 --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

echo "=========================================="
echo "All R3 evals complete."
echo "=========================================="

# ── Update master metrics ─────────────────────────────────────────────────────
echo "Updating master metrics..."
python scripts/rsl_rl/collect_rollout_metrics.py

echo ""
echo "Launch dashboard with:"
echo "  python scripts/rsl_rl/dashboard.py"
