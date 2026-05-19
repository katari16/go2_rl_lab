#!/bin/bash
#SBATCH --job-name=r10-priv-norand
#SBATCH --output=slurm_logs/r10_priv_norand_%j.out
#SBATCH --error=slurm_logs/r10_priv_norand_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── R10: R4 (no rand) + privileged estimator — absolute capacity ceiling ─────
# Combines R4's noise-free training (no mass rand, no pushes, no obs noise) with
# R9's privileged estimator inputs (mass + lin_vel + contacts). If R10's error
# stays well above zero, the estimator architecture itself is the bottleneck
# rather than the data observability.

module load eth_proxy

TASK="Go2-Ablation-R10-v0"

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
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
