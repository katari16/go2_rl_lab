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

run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-A1-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_analysis/batch1/batch1_ablation/ablation_A1_h10_3d_rec/2026-04-06_23-39-14/model_16000.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-A2-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_analysis/batch1/batch1_ablation/ablation_A2_h40_3d_rec/2026-04-06_23-39-11/model_16000.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-C1-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_analysis/batch1/batch1_ablation/ablation_C1_h10_3d_norec/2026-04-06_23-39-14/model_16500.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-C2-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_analysis/batch1/batch1_ablation/ablation_C2_h40_3d_norec/2026-04-06_23-39-15/model_16000.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-B2-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_analysis/batch1/batch1_ablation/ablation_B2_h40_6d_rec/2026-04-06_23-39-11/model_16000.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

# NOTE: B3 only reached model_9500 (training stopped early)
run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-B3-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_analysis/batch1/batch1_ablation/ablation_B3_h40_6d_rec_big/2026-04-06_23-39-14/model_16500.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-J1-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_force_accuracy_reward/ablation_J1_4d_h40_estrew_w1_30N/2026-04-15_17-24-50/model_17000.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-J3-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_force_accuracy_reward/ablation_J3_4d_h40_estrew_w50_30N/2026-04-15_17-24-50/model_17000.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

run_eval "python scripts/rsl_rl/eval/rollout_estimator_eval.py --task Go2-Ablation-J5-v0 --checkpoint /home/ubuntu/go2_rl_lab/logs/rsl_rl/ablation_force_accuracy_reward/ablation_J5_4d_h40_estrew_w50_tcnpre_30N/2026-04-15_17-24-57/model_16000.pt --num_envs 128 --training_regime --duration 10 --no_active_mask --headless"

echo "=========================================="
echo "All rollout estimator evals complete."
echo "=========================================="
