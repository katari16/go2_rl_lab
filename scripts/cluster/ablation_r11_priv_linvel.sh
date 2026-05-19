#!/bin/bash
#SBATCH --job-name=r11-priv-linvel
#SBATCH --output=slurm_logs/r11_priv_linvel_%j.out
#SBATCH --error=slurm_logs/r11_priv_linvel_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── R11: R1 + privileged base linear velocity only (peel-back) ───────────────
# Same as R9 but only the GT base lin_vel (3) is added to the estimator input.
# Compared to R9, isolates how much of the privileged-info gain comes from
# direct knowledge of the base acceleration term in Newton's 2nd law.

module load eth_proxy

TASK="Go2-Ablation-R11-v0"

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
