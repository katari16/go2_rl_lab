#!/bin/bash
#SBATCH --job-name=rollout-eval-h1h8h9
#SBATCH --output=slurm_logs/rollout_eval_h1h8h9_%A_%a.out
#SBATCH --error=slurm_logs/rollout_eval_h1h8h9_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=0-5

# Rollout estimator evaluation for H1, H8, H9 ablation runs.
# SLURM job array — one GPU per run, 6 parallel.
#
# 50N variants:  --force_baskets 10 20 30 40 50
# 100N variants: --force_baskets 20 40 60 80 100
#
# Usage:
#   sbatch scripts/cluster/rollout_eval_h1_h8_h9.sh

module load eth_proxy

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab

EVAL_CMD="python scripts/rsl_rl/eval/rollout_estimator_eval.py"
COMMON="--num_envs 4096 --duration 20 --headless"
BASKETS_50N="--force_baskets 10 20 30 40 50"
BASKETS_100N="--force_baskets 20 40 60 80 100"

TASKS=(
  "Go2-Ablation-H1-50N-v0"
  "Go2-Ablation-H1-100N-v0"
  "Go2-Ablation-H8-50N-v0"
  "Go2-Ablation-H8-100N-v0"
  "Go2-Ablation-H9-50N-v0"
  "Go2-Ablation-H9-100N-v0"
)

CHECKPOINTS=(
  "logs/rsl_rl/ablation_H1_3d_h30_50N/2026-04-08_22-05-44/model_12500.pt"
  "logs/rsl_rl/ablation_H1_3d_h30_100N/2026-04-08_22-06-23/model_13500.pt"
  "logs/rsl_rl/ablation_H8_3d_h30_estrew_50N/2026-04-08_22-05-41/model_14000.pt"
  "logs/rsl_rl/ablation_H8_3d_h30_estrew_100N/2026-04-08_22-05-41/model_14500.pt"
  "logs/rsl_rl/ablation_H9_2d_h30_50N/2026-04-08_22-05-41/model_14000.pt"
  "logs/rsl_rl/ablation_H9_2d_h30_100N/2026-04-08_22-05-41/model_14500.pt"
)

# Even indices (0,2,4) are 50N, odd indices (1,3,5) are 100N
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
CKPT=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
  BASKETS=$BASKETS_50N
else
  BASKETS=$BASKETS_100N
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
