#!/bin/bash
# Unified eval pipeline for the randomization comparison (R1, R3, R4).
#
#   R1: full domain randomization (deployed baseline reference)
#   R3: same as R1 but mass randomization disabled
#   R4: same as R1 but mass + pushes + observation noise all disabled
#
# For each run, executes three evaluations using the latest model checkpoint:
#   1. static_eval.py           — 12 envs, --save_data (writes time-series JSON
#                                 used by the appendix plots)
#   2. ou_force_eval.py         — 4096 envs, 20s OU disturbance; metrics.json
#                                 contains mae±std, per-axis mae±std, etc.
#   3. rollout_estimator_eval.py — 4096 envs, 20s training_regime rollout;
#                                  metrics.json mirrors the OU output schema.
#
# Outputs land in:
#   data/eval/<experiment_name>/static_<timestamp>/
#   data/eval/<experiment_name>/ou_force_eval_<timestamp>/
#   data/eval/<experiment_name>/rollout_estimator_<timestamp>_..._training_regime/
#
# Tip: tail one of the slurm logs while running to monitor.

set -e
cd "$(dirname "$0")/../.."

BASE=/home/ubuntu/go2_rl_lab/logs/rsl_rl

R1_CKPT=$BASE/ablations_p_series/ablation_R1_h30_6d_big_tcn_norec/2026-04-21_19-15-56/model_9500.pt
R3_CKPT=$BASE/ablation_R3_h30_6d_big_tcn_norec_nomassrand/2026-04-23_23-09-49/model_10000.pt
R4_CKPT=$BASE/ablations_p_series/ablation_R4_h30_6d_big_tcn_norec_norand/2026-05-16_09-50-13/model_9000.pt

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
echo "Randomization comparison eval — start: $(date)"
echo "=========================================="

run_suite "Go2-Ablation-R1-v0" "$R1_CKPT"
run_suite "Go2-Ablation-R3-v0" "$R3_CKPT"
run_suite "Go2-Ablation-R4-v0" "$R4_CKPT"

echo ""
echo "=========================================="
echo "All evals complete — $(date)"
echo "=========================================="
