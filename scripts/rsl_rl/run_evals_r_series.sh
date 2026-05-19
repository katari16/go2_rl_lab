#!/bin/bash
# Full eval pipeline for R-series (deployed estimator candidates):
#   R1: h30, 6D, big net, TCN preprocessor, no rec loss
#   R2: h30, 6D, big net, TCN preprocessor, no rec loss, est-acc reward
#   R3: h30, 6D, big net, TCN preprocessor, no rec loss, no mass randomization
#
# Each run: static_eval (12 envs), ou_force_eval (4096 envs, 20s),
#           rollout_estimator_eval (4096 envs, 20s, training_regime).

set -e
cd "$(dirname "$0")/../.."

BASE_P=/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series

R1=$BASE_P/ablation_R1_h30_6d_big_tcn_norec/2026-04-21_19-15-56/model_9500.pt
R2=$BASE_P/ablation_R2_h30_6d_big_tcn_norec_estacc/2026-04-21_19-16-02/model_9500.pt
R3=/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_R3_h30_6d_big_tcn_norec_nomassrand/2026-04-23_23-09-49/model_10000.pt

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
echo "R-series eval — start: $(date)"
echo "=========================================="

run_suite "Go2-Ablation-R1-v0" "$R1"
run_suite "Go2-Ablation-R2-v0" "$R2"
run_suite "Go2-Ablation-R3-v0" "$R3"

echo ""
echo "=========================================="
echo "All R-series evals complete — $(date)"
echo "=========================================="
