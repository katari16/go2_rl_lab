#!/bin/bash
#SBATCH --job-name=go2-lowlevel-no-est
#SBATCH --output=slurm_logs/lowlevel_no_est_%j.out
#SBATCH --error=slurm_logs/lowlevel_no_est_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── Low-level walking policy WITHOUT force estimate in obs ───────────
# 57-dim policy obs (proprioceptive only), 67-dim critic obs.
# Standard OnPolicyRunner, forces active from start at 20N.
# After convergence, freeze this policy and train estimator on top.

module load eth_proxy

TASK="Go2-LowLevel-NoEst-v0"

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
