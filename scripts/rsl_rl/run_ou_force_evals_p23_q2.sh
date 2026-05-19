#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

run_eval() {
    echo "=========================================="
    echo "Running: $1"
    echo "=========================================="
    eval "$1"
    echo ""
}

BASE=/home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P23-v0 --checkpoint $BASE/ablation_P23_h4_4d_paint_trap/2026-04-19_17-43-55/model_11000.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P24-v0 --checkpoint $BASE/ablation_P24_h8_4d_paint_trap/2026-04-19_17-43-39/model_12000.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P25-v0 --checkpoint $BASE/ablation_P25_h10_4d_paint_trap/2026-04-19_17-45-43/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P26-v0 --checkpoint $BASE/ablation_P26_h8_4d_paint_trap_norec/2026-04-19_17-45-43/model_12000.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P27-v0 --checkpoint $BASE/ablation_P27_h8_4d_paint_trap_zero20/2026-04-19_17-45-43/model_12000.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P28-v0 --checkpoint $BASE/ablation_P28_h8_4d_constant/2026-04-19_17-45-43/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P29-v0 --checkpoint $BASE/ablation_P29_h8_4d_cyclic_trap/2026-04-19_17-45-43/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P30-v0 --checkpoint $BASE/ablation_P30_h4_4d_constant/2026-04-19_17-45-02/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P31-v0 --checkpoint $BASE/ablation_P31_h8_4d_constant_b/2026-04-19_17-45-02/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P32a-v0 --checkpoint $BASE/ablation_P32a_h8_4d_paint_trap_comp_b30_s025/2026-04-19_17-45-02/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P32b-v0 --checkpoint $BASE/ablation_P32b_h8_4d_paint_trap_comp_b30_s05/2026-04-19_17-45-02/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P32c-v0 --checkpoint $BASE/ablation_P32c_h8_4d_paint_trap_comp_b40_s025/2026-04-19_17-45-02/model_12000.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P32d-v0 --checkpoint $BASE/ablation_P32d_h8_4d_paint_trap_comp_b40_s05/2026-04-19_17-45-02/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P33-v0 --checkpoint $BASE/ablation_P33_h8_4d_paint_trap_estacc/2026-04-19_17-45-02/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P34-v0 --checkpoint $BASE/ablation_P34_h8_4d_paint_trap_both_b30_s025/2026-04-19_17-45-02/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P35-v0 --checkpoint $BASE/ablation_P35_h8_4d_paint_trap_estacc_tcn/2026-04-19_17-45-37/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-Q1-v0 --checkpoint $BASE/ablation_Q1_h10_4d_paint_trap_big_tcn_norec_estacc/2026-04-20_02-23-32/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"
run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-Q2-v0 --checkpoint $BASE/ablation_Q2_h30_4d_paint_trap_big_tcn_norec_estacc/2026-04-20_02-21-12/model_12500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

echo "=========================================="
echo "All P23-Q2 OU force evals complete."
echo "=========================================="
