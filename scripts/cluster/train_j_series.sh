#!/bin/bash
#SBATCH --job-name=ablation-j
#SBATCH --output=slurm_logs/ablation_j_%A_%a.out
#SBATCH --error=slurm_logs/ablation_j_%A_%a.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=0-7

# J-series: 4D wrench + estimation accuracy reward sweep (30N, 10Nm yaw)
# CompliantOnPolicyRunner, h=40, default network
#
# | Idx | ID | Weight | Sigma | TCN        | Rec loss |
# |-----|----|--------|-------|------------|----------|
# |  0  | J1 | 1      | 1.0   | no         | yes      |
# |  1  | J2 | 10     | 1.0   | no         | yes      |
# |  2  | J3 | 50     | 1.0   | no         | yes      |
# |  3  | J4 | 100    | 1.0   | no         | yes      |
# |  4  | J5 | 50     | 1.0   | preprocess | yes      |
# |  5  | J6 | 50     | 1.0   | no         | no       |
# |  6  | J7 | 50     | 0.25  | no         | yes      |
# |  7  | J8 | 50     | 4.0   | no         | yes      |
#
# Usage:
#   sbatch scripts/cluster/train_j_series.sh

module load eth_proxy

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab

TASKS=(
  "Go2-Ablation-J1-v0"
  "Go2-Ablation-J2-v0"
  "Go2-Ablation-J3-v0"
  "Go2-Ablation-J4-v0"
  "Go2-Ablation-J5-v0"
  "Go2-Ablation-J6-v0"
  "Go2-Ablation-J7-v0"
  "Go2-Ablation-J8-v0"
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
