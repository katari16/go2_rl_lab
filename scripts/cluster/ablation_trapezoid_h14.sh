#!/bin/bash
#SBATCH --job-name=est-ablation-h14-h15
#SBATCH --output=slurm_logs/ablation_h14_h15_%a_%j.out
#SBATCH --error=slurm_logs/ablation_h14_h15_%a_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-2

# ── Force Estimator Ablation: H14 (trapezoid) + H15 (H3a rerun with fixes) ─
# Both: 6D, h=40, bigger net, yaw+tq_angle loss, 50N
# H14: PAINT-style trapezoid wrench + stratified magnitude buckets
# H15: constant wrench (same as H3a but with yaw loss + torque gating fixed)

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-H14-50N-v0" ;;
    2) TASK="Go2-Ablation-H15-50N-v0" ;;
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
