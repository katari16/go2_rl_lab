#!/bin/bash
#SBATCH --job-name=q1-q2
#SBATCH --output=slurm_logs/q1_q2_%a_%j.out
#SBATCH --error=slurm_logs/q1_q2_%a_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-2

# ── Q1-Q2 Ablation Study ─────────────────────────────────────────────────────
# Q1: PAINT trap, h=10, bigger net [256,128]/[64,32], TCN pre, no rec, est_acc w=50
# Q2: PAINT trap, h=30, bigger net [256,128]/[64,32], TCN pre, no rec, est_acc w=50
#
# All: 30N force, 10Nm torque, 4D estimator

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1)  TASK="Go2-Ablation-Q1-v0" ;;
    2)  TASK="Go2-Ablation-Q2-v0" ;;
    *) echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Task: $TASK"
echo "========================================="

nvidia-smi

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab
python scripts/rsl_rl/train.py --task ${TASK} --num_envs 4096 --headless

echo "Job ${TASK} completed at $(date)"
