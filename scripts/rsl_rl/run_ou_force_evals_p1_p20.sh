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

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P20-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P20_h30_4d_pd25/2026-04-19_11-10-59/model_13000.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P19-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P19_h30_4d_torque5nm/2026-04-19_11-11-07/model_12500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P18-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P18_h30_4d_payload/2026-04-19_11-11-07/model_11000.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P17-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P17_h30_6d_big/2026-04-19_11-11-07/model_12000.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P16-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P16_h30_6d/2026-04-19_11-11-07/model_12500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P15-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P15_h30_4d/2026-04-19_11-11-07/model_12500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P14-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P14_h30_xy_yaw/2026-04-19_11-11-07/model_12500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P13-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P13_h30_2d/2026-04-19_11-11-07/model_12500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P12-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P12_h30_4d_tcn/2026-04-19_11-11-03/model_12500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P11-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P11_h30_4d_norec/2026-04-19_11-11-17/model_12500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P10-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P10_h30_4d_comp_w5p0/2026-04-19_11-11-36/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P9-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P9_h30_4d_comp_w1p0/2026-04-19_11-11-36/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P8-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P8_h30_4d_comp_w0p5/2026-04-19_11-11-32/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P7-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P7_h30_4d_estrew_w50/2026-04-19_11-11-32/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P6-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P6_h30_4d_double/2026-04-19_11-11-32/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P5-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P5_h30_4d_half/2026-04-19_11-12-50/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P4-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P4_h40_4d/2026-04-19_11-11-09/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P3-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P3_h30_4d/2026-04-19_11-11-09/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P2-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P2_h20_4d/2026-04-19_11-12-59/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

run_eval "python scripts/rsl_rl/eval/ou_force_eval.py --task Go2-Ablation-P1-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablations_p_series/ablation_P1_h10_4d/2026-04-19_11-13-01/model_11500.pt --num_envs 12 --show_est --show_gt --real-time --duration 20 --headless"

echo "=========================================="
echo "All OU force evals complete."
echo "=========================================="
