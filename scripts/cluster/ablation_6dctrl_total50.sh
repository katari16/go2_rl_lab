#!/bin/bash
#SBATCH --job-name=6dctrl-t50
#SBATCH --output=slurm_logs/6dctrl_t50_%a_%j.out
#SBATCH --error=slurm_logs/6dctrl_t50_%a_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-4

# ── 6Dctrl Total50 family ───────────────────────────────────────────────────
# All four share: R1-arch estimator (6D, enc=[256,128], TCN pre, no rec),
# pose commands (roll/pitch/height), force_gate_mode="total", threshold=50.
# Differ only in the force_est_accuracy reward weight:
#   1) Total50         — no est-acc reward
#   2) Total50-EstAccW10 — w=10, sigma=1.0
#   3) Total50-EstAccW25 — w=25, sigma=1.0
#   4) Total50-EstAccW50 — w=50, sigma=1.0 (matches R2)

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-6Dctrl-Total50-v0" ;;
    2) TASK="Go2-Ablation-6Dctrl-Total50-EstAccW10-v0" ;;
    3) TASK="Go2-Ablation-6Dctrl-Total50-EstAccW25-v0" ;;
    4) TASK="Go2-Ablation-6Dctrl-Total50-EstAccW50-v0" ;;
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
