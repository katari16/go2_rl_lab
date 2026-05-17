#!/bin/bash
# Full eval pipeline for R4 (observability study — no mass rand, no push, no obs noise).
#
# Runs the same three evals as the R-series script so the results are directly
# comparable to R1/R3 in the report:
#   - static_eval: 12 envs, --save_data dumps the time-series JSON used by the
#                  thesis appendix plots.
#   - ou_force_eval: 4096 envs, 20s; metrics.json has force_mae±std,
#                    torque_mae±std, per_axis_mae±std, angular_err mean/median±std.
#   - rollout_estimator_eval: 4096 envs, 20s, training_regime; same fields plus
#                             relative_err_pct.

set -e
cd "$(dirname "$0")/../.."

R4=/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_R4_h30_6d_big_tcn_norec_norand/2026-05-16_09-50-13/model_9000.pt

if [ ! -f "$R4" ]; then
    echo "!! R4 checkpoint not found: $R4"
    exit 1
fi

run_eval() {
    echo "------------------------------------------"
    echo "Running: $1"
    echo "------------------------------------------"
    eval "$1"
    echo ""
}

echo "=========================================="
echo "R4 eval — start: $(date)"
echo "Checkpoint: $R4"
echo "=========================================="

run_eval "python scripts/rsl_rl/static_eval.py --task Go2-Ablation-R4-v0 --checkpoint $R4 --num_envs 12 --show_est --force_min 10.0 --force_max 30.0 --save_data --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-R4-v0 --checkpoint $R4 --num_envs 4096 --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-R4-v0 --checkpoint $R4 --num_envs 4096 --duration 20 --training_regime --no_active_mask --headless"

echo ""
echo "=========================================="
echo "R4 evals complete — $(date)"
echo "=========================================="
