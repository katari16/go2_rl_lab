#!/bin/bash
#SBATCH --job-name=ablation-k
#SBATCH --output=slurm_logs/ablation_k_%A_%a.out
#SBATCH --error=slurm_logs/ablation_k_%A_%a.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=0-2

# K-series: B4 re-run (yaw bug fixed) + est accuracy reward (20N, 5Nm, 6D)
#
# | Idx | ID | Est-acc weight | Notes                      |
# |-----|----|----------------|----------------------------|
# |  0  | K1 | —              | B4 re-run, yaw bug fixed   |
# |  1  | K2 | 50             | + est accuracy reward      |
# |  2  | K3 | 100            | + est accuracy reward      |
#
# Usage:
#   sbatch scripts/cluster/train_k_series.sh

module load eth_proxy

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab

TASKS=(
  "Go2-Ablation-K1-v0"
  "Go2-Ablation-K2-v0"
  "Go2-Ablation-K3-v0"
)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID (array: $SLURM_ARRAY_TASK_ID)"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Task: $TASK"
echo "========================================="

nvidia-smi

python scripts/rsl_rl/train.py --task $TASK --num_envs 4096 --headless

echo "Completed $TASK at $(date)"
