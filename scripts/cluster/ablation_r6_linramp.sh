#!/bin/bash
#SBATCH --job-name=r6-linramp
#SBATCH --output=slurm_logs/r6_linramp_%j.out
#SBATCH --error=slurm_logs/r6_linramp_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── R6: R1 with linear force ramp post-gate (10→30N over 2500 iters) ─────────
# Same R1 env and estimator. Reward gate fires as usual (~iter 2100);
# at that moment force_range = (0, 10) and ramps linearly to (0, 30) over
# 2500 iterations. Torque max scales proportionally.

module load eth_proxy

TASK="Go2-Ablation-R6-v0"

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
