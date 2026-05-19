#!/bin/bash
# Eval pipeline for the force curriculum comparison (R6, R8).
#
#   R6: R1 baseline + linear force ramp post-gate (10→30N over 2500 iters)
#   R8: R1 baseline + bucketed force curriculum post-gate (10/20/30N × 1000 iters)
#
# Use alongside scripts/rsl_rl/run_evals_r1_r3_r4.sh so R1's evals are available
# as the hard-step reference for the curriculum comparison.
#
# For each run, executes:
#   1. static_eval.py            — 12 envs, --save_data (writes time-series JSON)
#   2. ou_force_eval.py          — 4096 envs, 20s OU disturbance
#   3. rollout_estimator_eval.py — 4096 envs, 20s training_regime rollout
#
# Outputs land in go2_rl_lab/data/eval/<experiment_name>/.

set -e
cd "$(dirname "$0")/../.."

BASE=/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series

R6_CKPT=$BASE/ablation_R6_h30_6d_big_tcn_norec_linramp/2026-05-17_12-53-33/model_10000.pt
R8_CKPT=$BASE/ablation_R8_h30_6d_big_tcn_norec_buckets/2026-05-17_12-53-33/model_10000.pt

run_eval() {
    echo "------------------------------------------"
    echo "Running: $1"
    echo "------------------------------------------"
    eval "$1"
    echo ""
}

run_suite() {
    local task="$1"
    local ckpt="$2"
    echo ""
    echo "##########################################"
    echo "  $task    (ckpt: $(basename $ckpt))"
    echo "##########################################"

    if [ ! -f "$ckpt" ]; then
        echo "!! Checkpoint not found: $ckpt — skipping"
        return
    fi

    run_eval "python scripts/rsl_rl/static_eval.py --task $task --checkpoint $ckpt --num_envs 12 --show_est --force_min 10.0 --force_max 30.0 --save_data --headless"

    run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task $task --checkpoint $ckpt --num_envs 4096 --duration 20 --headless"

    run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task $task --checkpoint $ckpt --num_envs 4096 --duration 20 --training_regime --no_active_mask --headless"
}

echo "=========================================="
echo "Force curriculum eval (R6, R8) — start: $(date)"
echo "=========================================="

run_suite "Go2-Ablation-R6-v0" "$R6_CKPT"
run_suite "Go2-Ablation-R8-v0" "$R8_CKPT"

echo ""
echo "=========================================="
echo "R6/R8 evals complete — $(date)"
echo "=========================================="
