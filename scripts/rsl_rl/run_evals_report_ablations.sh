#!/bin/bash
# Full eval pipeline for the report-section ablations:
#   - History sweep:   P1, P2, P3, P4
#   - Network size:    P5, P3, P6   (4D)  and  P16, P17  (6D)
#   - Rec loss:        P3 vs P11    (no est-acc reward),  J3 vs J6  (with est-acc reward)
#   - Force dim:       P13 (2D), P14 (xy_yaw), P3 (4D), P16 (6D), P17 (6D big)
#   - TCN:             J3 vs J5
#   - PD gains:        P3 (Kp=8) vs P20 (Kp=25)
#
# Each run: static_eval (12 envs), ou_force_eval (4096 envs, 20s),
#           rollout_estimator_eval (4096 envs, training_regime).
#
# Overnight run — exits on first failure.

set -e
cd "$(dirname "$0")/../.."

BASE_P=/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series
BASE_J=/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_force_accuracy_reward

# ── P-series checkpoints (latest model_*.pt under each run dir) ──────────────
P1=$BASE_P/ablation_P1_h10_4d/2026-04-19_11-13-01/model_11500.pt
P2=$BASE_P/ablation_P2_h20_4d/2026-04-19_11-12-59/model_11500.pt
P3=$BASE_P/ablation_P3_h30_4d/2026-04-19_11-11-09/model_11500.pt
P4=$BASE_P/ablation_P4_h40_4d/2026-04-19_11-11-09/model_11500.pt
P5=$BASE_P/ablation_P5_h30_4d_half/2026-04-19_11-12-50/model_11500.pt
P6=$BASE_P/ablation_P6_h30_4d_double/2026-04-19_11-11-32/model_11500.pt
P11=$BASE_P/ablation_P11_h30_4d_norec/2026-04-19_11-11-17/model_12500.pt
P13=$BASE_P/ablation_P13_h30_2d/2026-04-19_11-11-07/model_12500.pt
P14=$BASE_P/ablation_P14_h30_xy_yaw/2026-04-19_11-11-07/model_12500.pt
P16=$BASE_P/ablation_P16_h30_6d/2026-04-19_11-11-07/model_12500.pt
P17=$BASE_P/ablation_P17_h30_6d_big/2026-04-19_11-11-07/model_12000.pt
P20=$BASE_P/ablation_P20_h30_4d_pd25/2026-04-19_11-10-59/model_13000.pt

# ── J-series checkpoints ─────────────────────────────────────────────────────
J3=$BASE_J/ablation_J3_4d_h40_estrew_w50_30N/2026-04-15_17-24-50/model_17000.pt
J5=$BASE_J/ablation_J5_4d_h40_estrew_w50_tcnpre_30N/2026-04-15_17-24-57/model_16000.pt
J6=$BASE_J/ablation_J6_4d_h40_estrew_w50_norec_30N/2026-04-15_17-24-57/model_17000.pt


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

    run_eval "python scripts/rsl_rl/static_eval.py \
        --task $task --checkpoint $ckpt \
        --num_envs 12 --show_est \
        --force_min 10.0 --force_max 30.0 --save_data --headless"

    run_eval "python scripts/rsl_rl/eval/ou_force_eval.py \
        --task $task --checkpoint $ckpt \
        --num_envs 4096 --duration 20 --headless"

    run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py \
        --task $task --checkpoint $ckpt \
        --num_envs 4096 --duration 20 \
        --training_regime --no_active_mask --headless"
}


echo "=========================================="
echo "Report ablations eval — start: $(date)"
echo "=========================================="

# ── 1. History sweep ─────────────────────────────────────────────────────────
run_suite "Go2-Ablation-P1-v0"  "$P1"
run_suite "Go2-Ablation-P2-v0"  "$P2"
run_suite "Go2-Ablation-P3-v0"  "$P3"
run_suite "Go2-Ablation-P4-v0"  "$P4"

# ── 2. Network size sweep (4D) ───────────────────────────────────────────────
run_suite "Go2-Ablation-P5-v0"  "$P5"
# P3 already run above (baseline)
run_suite "Go2-Ablation-P6-v0"  "$P6"

# ── 3. Force dim sweep ───────────────────────────────────────────────────────
run_suite "Go2-Ablation-P13-v0" "$P13"
run_suite "Go2-Ablation-P14-v0" "$P14"
# P3 already run (4D baseline)
run_suite "Go2-Ablation-P16-v0" "$P16"
run_suite "Go2-Ablation-P17-v0" "$P17"

# ── 4. Rec loss — P series (no est-acc reward) ───────────────────────────────
run_suite "Go2-Ablation-P11-v0" "$P11"
# P3 already run (with rec)

# ── 5. PD gains ──────────────────────────────────────────────────────────────
run_suite "Go2-Ablation-P20-v0" "$P20"

# ── 6. TCN + Rec loss — J series ─────────────────────────────────────────────
run_suite "Go2-Ablation-J3-v0"  "$J3"
run_suite "Go2-Ablation-J5-v0"  "$J5"
run_suite "Go2-Ablation-J6-v0"  "$J6"


echo ""
echo "=========================================="
echo "All report ablation evals complete — $(date)"
echo "=========================================="
