#!/bin/bash
#SBATCH --job-name=rollout-eval
#SBATCH --output=slurm_logs/rollout_eval_%A_%a.out
#SBATCH --error=slurm_logs/rollout_eval_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=0-15

# Rollout estimator evaluation — SLURM job array (one GPU per run).
#
# 20N series (A/B/C/E): --force_baskets 5 10 15 20
# 50N series (H):       --force_baskets 10 20 30 40 50
#
# Usage:
#   sbatch scripts/cluster/rollout_eval_all.sh

module load eth_proxy

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab

EVAL_CMD="python scripts/rsl_rl/eval/rollout_estimator_eval.py"
COMMON="--num_envs 4096 --duration 20 --headless"
BASKETS_20N="--force_baskets 5 10 15 20"
BASKETS_50N="--force_baskets 10 20 30 40 50"

# ── Run table: task, checkpoint, baskets ──────────────────────────────
TASKS=(
  "Go2-Ablation-A1-v0"
  "Go2-Ablation-A2-v0"
  "Go2-Ablation-B2-v0"
  "Go2-Ablation-B3-v0"
  "Go2-Ablation-C1-v0"
  "Go2-Ablation-C2-v0"
  "Go2-Ablation-E1-v0"
  "Go2-Ablation-E2-v0"
  "Go2-Ablation-H12b-50N-v0"
  "Go2-Ablation-H13b-50N-v0"
  "Go2-Ablation-H14-50N-v0"
  "Go2-Ablation-H15-50N-v0"
  "Go2-Ablation-H16-50N-v0"
  "Go2-Ablation-H17-50N-v0"
  "Go2-Ablation-H18-50N-v0"
  "Go2-Ablation-H19-50N-v0"
)

CHECKPOINTS=(
  "logs/rsl_rl/ablation_A1_h10_3d_rec/2026-04-06_23-39-14/model_16000.pt"
  "logs/rsl_rl/ablation_A2_h40_3d_rec/2026-04-06_23-39-11/model_16000.pt"
  "logs/rsl_rl/ablation_B2_h40_6d_rec/2026-04-06_23-39-11/model_16000.pt"
  "logs/rsl_rl/ablation_B3_h40_6d_rec_big/2026-04-06_23-39-14/model_16500.pt"
  "logs/rsl_rl/ablation_C1_h10_3d_norec/2026-04-06_23-39-14/model_16500.pt"
  "logs/rsl_rl/ablation_C2_h40_3d_norec/2026-04-06_23-39-15/model_16000.pt"
  "logs/rsl_rl/ablation_E1_h20_3d_compliance/2026-04-06_23-39-15/model_16000.pt"
  "logs/rsl_rl/ablation_E2_h20_4d_compliance/2026-04-06_23-39-11/model_17500.pt"
  "logs/rsl_rl/ablation_H12b_6d_h40_tcnrep_50N/2026-04-09_22-50-46/model_12500.pt"
  "logs/rsl_rl/ablation_H13b_4d_h30_tcnrep_50N/2026-04-09_23-01-22/model_12500.pt"
  "logs/rsl_rl/ablation_H14_6d_h40_big_yaw_trap_50N/2026-04-10_20-45-47/model_12500.pt"
  "logs/rsl_rl/ablation_H15_6d_h40_big_yaw_fixed_50N/2026-04-10_20-45-47/model_12500.pt"
  "logs/rsl_rl/ablation_H16_6d_h40_big_equal_50N/2026-04-12_22-57-57/model_17000.pt"
  "logs/rsl_rl/ablation_H17_6d_h40_big_lindecay_50N/2026-04-12_22-57-57/model_17000.pt"
  "logs/rsl_rl/ablation_H18_6d_h40_big_tcnpre_detach_50N/2026-04-12_22-57-57/model_16500.pt"
  "logs/rsl_rl/ablation_H19_xy_yaw_h40_big_50N/2026-04-12_22-57-57/model_17000.pt"
)

# Indices 0-7 are 20N, 8-15 are 50N
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
CKPT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

if [ $SLURM_ARRAY_TASK_ID -lt 8 ]; then
  BASKETS=$BASKETS_20N
else
  BASKETS=$BASKETS_50N
fi

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID (array: $SLURM_ARRAY_TASK_ID)"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Task: $TASK"
echo "Checkpoint: $CKPT"
echo "Baskets: $BASKETS"
echo "========================================="

nvidia-smi

$EVAL_CMD --task $TASK --checkpoint $CKPT $COMMON $BASKETS

echo "Completed $TASK at $(date)"
