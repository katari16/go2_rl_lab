#!/bin/bash
#SBATCH --job-name=px-series
#SBATCH --output=slurm_logs/px_series_%a_%j.out
#SBATCH --error=slurm_logs/px_series_%a_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-2

# ── PX-series: PhysX direct wrench ablations ──────────────────────────────
# PX1: constant wrench (h30, 4D, 30N/10Nm, uniform, fz_scale=0.8)
# PX2: trapezoid wrench (h30, 4D, 30N/10Nm, uniform, fz_scale=0.8)

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-PX1-v0" ;;
    2) TASK="Go2-Ablation-PX2-v0" ;;
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
python scripts/rsl_rl/train.py \
    --task ${TASK} \
    --num_envs 4096 \
    --headless

echo "Job ${TASK} completed at $(date)"
