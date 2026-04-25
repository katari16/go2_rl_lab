#!/bin/bash
#SBATCH --job-name=6dctrl-curr
#SBATCH --output=slurm_logs/6dctrl_curr_%a_%j.out
#SBATCH --error=slurm_logs/6dctrl_curr_%a_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-2

# ── 6Dctrl force-gate curriculum ablations ───────────────────────────────────
# (1) Excluded: sum all rewards EXCEPT track_roll_pitch + track_height,
#     trigger when sum >= 30 (R1 baseline).
# (2) Tracking: trigger when ALL tracking rewards sustain their per-channel
#     percentage thresholds for >=50 consecutive iterations.

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1)  TASK="Go2-Ablation-6Dctrl-Excluded-v0" ;;
    2)  TASK="Go2-Ablation-6Dctrl-Tracking-v0" ;;
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
