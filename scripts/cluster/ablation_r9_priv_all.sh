#!/bin/bash
#SBATCH --job-name=r9-priv-all
#SBATCH --output=slurm_logs/r9_priv_all_%j.out
#SBATCH --error=slurm_logs/r9_priv_all_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── R9: R1 + privileged estimator (mass + base lin_vel + foot contacts) ──────
# Same R1 env (full randomization). Estimator receives 3 extra sets of inputs:
# GT robot mass (1), GT base linear velocity (3), GT foot contact force norms (4).
# Policy obs UNCHANGED so the policy cannot cheat. Quantifies the upper bound
# of estimator accuracy given more information under realistic training noise.

module load eth_proxy

TASK="Go2-Ablation-R9-v0"

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
