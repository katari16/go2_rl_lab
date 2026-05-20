#!/bin/bash
# Unified evaluation script for all report ablations.
#
# Runs three evaluation protocols per configuration:
#   1. static_eval.py            — 12 envs, constant forces, time-series JSON
#   2. ou_force_eval.py          — 4096 envs, 20s OU disturbance
#   3. rollout_estimator_eval.py — 4096 envs, 20s training-regime rollout
#
# Usage:
#   ./scripts/rsl_rl/run_eval.sh --task Go2-Ablation-R1-v0 --checkpoint <path>
#   ./scripts/rsl_rl/run_eval.sh --group architecture
#   ./scripts/rsl_rl/run_eval.sh --group randomization
#   ./scripts/rsl_rl/run_eval.sh --group curriculum
#   ./scripts/rsl_rl/run_eval.sh --group observability
#   ./scripts/rsl_rl/run_eval.sh --group deployed
#
# Groups correspond to report sections:
#   architecture   — History length, network capacity, TCN, reconstruction loss,
#                    wrench dimensionality, PD gains (Section 5.1)
#   randomization  — Domain randomization ablation (Section 5.1.6)
#   curriculum     — Force curriculum strategies (Section 5.1.6)
#   observability  — Privileged observation inputs (Section 5.1.6)
#   deployed       — Deployed configuration + payload (Section 5.2)

set -e
cd "$(dirname "$0")/../.."

NUM_ENVS=${NUM_ENVS:-4096}
DURATION=${DURATION:-20}

usage() {
    echo "Usage:"
    echo "  $0 --task <TASK_ID> --checkpoint <PATH>"
    echo "  $0 --group <GROUP_NAME>"
    echo ""
    echo "Groups: architecture, randomization, curriculum, observability, deployed"
    exit 1
}

run_suite() {
    local task="$1"
    local ckpt="$2"
    echo ""
    echo "##########################################"
    echo "  $task"
    echo "  checkpoint: $(basename $ckpt)"
    echo "##########################################"

    if [ ! -f "$ckpt" ]; then
        echo "!! Checkpoint not found: $ckpt — skipping"
        return
    fi

    echo "--- static_eval ---"
    python scripts/rsl_rl/static_eval.py --task $task --checkpoint $ckpt --num_envs 12 --show_est --force_min 10.0 --force_max 30.0 --save_data --headless

    echo "--- ou_force_eval ---"
    python scripts/rsl_rl/eval/ou_force_eval.py --task $task --checkpoint $ckpt --num_envs $NUM_ENVS --duration $DURATION --headless

    echo "--- rollout_estimator_eval (training_regime) ---"
    python scripts/rsl_rl/eval/rollout_estimator_eval.py --task $task --checkpoint $ckpt --num_envs $NUM_ENVS --duration $DURATION --training_regime --no_active_mask --headless
}

# ── Parse arguments ───────────────────────────────────────────────────────────

TASK=""
CHECKPOINT=""
GROUP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --num_envs) NUM_ENVS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ── Single task mode ──────────────────────────────────────────────────────────

if [ -n "$TASK" ] && [ -n "$CHECKPOINT" ]; then
    run_suite "$TASK" "$CHECKPOINT"
    echo ""
    echo "Done."
    exit 0
fi

# ── Group mode ────────────────────────────────────────────────────────────────

if [ -z "$GROUP" ]; then
    usage
fi

BASE=/home/ubuntu/go2_rl_lab/logs/rsl_rl
BASE_P=$BASE/ablations_p_series
BASE_J=$BASE/ablation_force_accuracy_reward

latest_ckpt() {
    ls "$1"/model_*.pt 2>/dev/null | sed 's/.*model_//' | sed 's/\.pt//' | sort -n | tail -1 | xargs -I{} echo "$1/model_{}.pt"
}

case $GROUP in
    architecture)
        echo "=========================================="
        echo "Architecture ablation (Report Section 5.1)"
        echo "=========================================="

        # History length: H=10, H=20, H=30, H=40
        run_suite "Go2-Ablation-P1-v0" "$BASE_P/ablation_P1_h10_4d/2026-04-19_11-13-01/model_11500.pt"
        run_suite "Go2-Ablation-P2-v0" "$BASE_P/ablation_P2_h20_4d/2026-04-19_11-12-59/model_11500.pt"
        run_suite "Go2-Ablation-P3-v0" "$BASE_P/ablation_P3_h30_4d/2026-04-19_11-11-09/model_11500.pt"
        run_suite "Go2-Ablation-P4-v0" "$BASE_P/ablation_P4_h40_4d/2026-04-19_11-11-09/model_11500.pt"

        # Network capacity: half, baseline, double
        run_suite "Go2-Ablation-P5-v0" "$BASE_P/ablation_P5_h30_4d_half/2026-04-19_11-12-50/model_11500.pt"
        run_suite "Go2-Ablation-P6-v0" "$BASE_P/ablation_P6_h30_4d_double/2026-04-19_11-11-32/model_11500.pt"

        # Wrench dimensionality: 2D, 3D (xy+yaw), 4D, 6D, 6D-big
        run_suite "Go2-Ablation-P13-v0" "$BASE_P/ablation_P13_h30_2d/2026-04-19_11-11-07/model_12500.pt"
        run_suite "Go2-Ablation-P14-v0" "$BASE_P/ablation_P14_h30_xy_yaw/2026-04-19_11-11-07/model_12500.pt"
        run_suite "Go2-Ablation-P16-v0" "$BASE_P/ablation_P16_h30_6d/2026-04-19_11-11-07/model_12500.pt"
        run_suite "Go2-Ablation-P17-v0" "$BASE_P/ablation_P17_h30_6d_big/2026-04-19_11-11-07/model_12000.pt"

        # Reconstruction loss: with vs without
        run_suite "Go2-Ablation-P11-v0" "$BASE_P/ablation_P11_h30_4d_norec/2026-04-19_11-11-17/model_12500.pt"

        # PD gains: Kp=8 vs Kp=25
        run_suite "Go2-Ablation-P20-v0" "$BASE_P/ablation_P20_h30_4d_pd25/2026-04-19_11-10-59/model_13000.pt"

        # TCN preprocessor + reconstruction loss (with est-accuracy reward)
        run_suite "Go2-Ablation-J3-v0" "$BASE_J/ablation_J3_4d_h40_estrew_w50_30N/2026-04-15_17-24-50/model_17000.pt"
        run_suite "Go2-Ablation-J5-v0" "$BASE_J/ablation_J5_4d_h40_estrew_w50_tcnpre_30N/2026-04-15_17-24-57/model_16000.pt"
        run_suite "Go2-Ablation-J6-v0" "$BASE_J/ablation_J6_4d_h40_estrew_w50_norec_30N/2026-04-15_17-24-57/model_17000.pt"
        ;;

    randomization)
        echo "=========================================="
        echo "Domain randomization ablation (Report Section 5.1.6)"
        echo "=========================================="
        run_suite "Go2-Ablation-R1-v0" "$BASE_P/ablation_R1_h30_6d_big_tcn_norec/2026-04-21_19-15-56/model_9500.pt"
        run_suite "Go2-Ablation-R3-v0" "$BASE/ablation_R3_h30_6d_big_tcn_norec_nomassrand/2026-04-23_23-09-49/model_10000.pt"
        run_suite "Go2-Ablation-R4-v0" "$BASE_P/ablation_R4_h30_6d_big_tcn_norec_norand/2026-05-16_09-50-13/model_9000.pt"
        ;;

    curriculum)
        echo "=========================================="
        echo "Force curriculum ablation (Report Section 5.1.6)"
        echo "=========================================="
        run_suite "Go2-Ablation-R1-v0" "$BASE_P/ablation_R1_h30_6d_big_tcn_norec/2026-04-21_19-15-56/model_9500.pt"
        run_suite "Go2-Ablation-R6-v0" "$BASE_P/ablation_R6_h30_6d_big_tcn_norec_linramp/2026-05-17_12-53-33/model_10000.pt"
        run_suite "Go2-Ablation-R8-v0" "$BASE_P/ablation_R8_h30_6d_big_tcn_norec_buckets/2026-05-17_12-53-33/model_10000.pt"
        ;;

    observability)
        echo "=========================================="
        echo "Privileged observations ablation (Report Section 5.1.6)"
        echo "=========================================="
        R9_DIR=$BASE_P/ablation_R9_h30_6d_big_tcn_norec_priv/2026-05-18_10-25-30
        R10_DIR=$BASE_P/ablation_R10_h30_6d_big_tcn_norec_priv_norand/2026-05-18_10-24-38
        R11_DIR=$BASE_P/ablation_R11_h30_6d_big_tcn_norec_priv_linvel/2026-05-18_10-24-38
        R12_DIR=$BASE_P/ablation_R12_h30_6d_big_tcn_norec_priv_contacts/2026-05-18_10-25-20

        run_suite "Go2-Ablation-R9-v0" "$(latest_ckpt $R9_DIR)"
        run_suite "Go2-Ablation-R10-v0" "$(latest_ckpt $R10_DIR)"
        run_suite "Go2-Ablation-R11-v0" "$(latest_ckpt $R11_DIR)"
        run_suite "Go2-Ablation-R12-v0" "$(latest_ckpt $R12_DIR)"
        ;;

    deployed)
        echo "=========================================="
        echo "Deployed configuration + payload (Report Section 5.2)"
        echo "=========================================="
        run_suite "Go2-Ablation-6Dctrl-Total50-v0" "$BASE/ablation_6Dctrl_curr_total50/2026-04-26_18-23-51/model_9500.pt"
        run_suite "Go2-Ablation-P18-v0" "$BASE/ablation_P18_h30_6d_big_tcn_norec_payload/2026-04-25_23-28-49/model_15000.pt"
        ;;

    *)
        echo "Unknown group: $GROUP"
        usage
        ;;
esac

echo ""
echo "=========================================="
echo "Evaluation complete — $(date)"
echo "=========================================="
