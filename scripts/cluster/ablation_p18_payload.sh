#!/bin/bash
#SBATCH --job-name=p18-payload
#SBATCH --output=slurm_logs/p18_payload_%j.out
#SBATCH --error=slurm_logs/p18_payload_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── P18: Payload (randomized 0-4kg mass per episode) — R1 architecture ──────
# 6D wrench estimator, enc=[256,128], TCN preprocessor, no rec loss, yaw+torque
# angle losses. Same as R1 but with a payload link whose mass varies 0-4kg per
# episode reset. Forces up to 30N / 10Nm from the P-series persistent_wrench.

module load eth_proxy

TASK="Go2-Ablation-P18-v0"

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
