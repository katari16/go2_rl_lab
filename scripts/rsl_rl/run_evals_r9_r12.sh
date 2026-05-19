#!/bin/bash
# Eval pipeline for the privileged estimator observability ablations (R9, R10, R11, R12).
#
#   R9:  Privileged (mass + lin_vel + contacts), full randomization
#   R10: Privileged (mass + lin_vel + contacts), no randomization
#   R11: Privileged (lin_vel only), full randomization
#   R12: Privileged (contacts only), full randomization
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

R9_DIR=$BASE/ablation_R9_h30_6d_big_tcn_norec_priv/2026-05-18_10-25-30
R10_DIR=$BASE/ablation_R10_h30_6d_big_tcn_norec_priv_norand/2026-05-18_10-24-38
R11_DIR=$BASE/ablation_R11_h30_6d_big_tcn_norec_priv_linvel/2026-05-18_10-24-38
R12_DIR=$BASE/ablation_R12_h30_6d_big_tcn_norec_priv_contacts/2026-05-18_10-25-20

latest_ckpt() {
    ls "$1"/model_*.pt 2>/dev/null | sed 's/.*model_//' | sed 's/\.pt//' | sort -n | tail -1 | xargs -I{} echo "$1/model_{}.pt"
}

R9_CKPT=$(latest_ckpt "$R9_DIR")
R10_CKPT=$(latest_ckpt "$R10_DIR")
R11_CKPT=$(latest_ckpt "$R11_DIR")
R12_CKPT=$(latest_ckpt "$R12_DIR")

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
echo "Privileged estimator eval (R9-R12) — start: $(date)"
echo "=========================================="
echo "  R9  ckpt: $R9_CKPT"
echo "  R10 ckpt: $R10_CKPT"
echo "  R11 ckpt: $R11_CKPT"
echo "  R12 ckpt: $R12_CKPT"
echo ""

run_suite "Go2-Ablation-R9-v0" "$R9_CKPT"
run_suite "Go2-Ablation-R10-v0" "$R10_CKPT"
run_suite "Go2-Ablation-R11-v0" "$R11_CKPT"
run_suite "Go2-Ablation-R12-v0" "$R12_CKPT"

echo ""
echo "=========================================="
echo "R9-R12 evals complete — $(date)"
echo "=========================================="
