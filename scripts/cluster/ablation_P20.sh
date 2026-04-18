#!/bin/bash
#SBATCH --job-name=ablation-p20
#SBATCH --output=slurm_logs/ablation_p20_%j.out
#SBATCH --error=slurm_logs/ablation_p20_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── PAINT Force Profile Ablation: P20 ──────────────────────────────────────
# P20: Default PD gains Kp=25, Kd=0.5 (baseline uses Kp=8, Kd=0.4)
# Same as P0 baseline but with higher stiffness/damping
#
# Forces: 30N max (XY), 25N (Z), torque 10Nm
# Estimator: h=30, 4D, default net [128,64]/[32,16], rec loss

module load eth_proxy

TASK="Go2-Ablation-P20-v0"

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
