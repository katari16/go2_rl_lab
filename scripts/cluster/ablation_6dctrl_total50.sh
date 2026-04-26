#!/bin/bash
#SBATCH --job-name=6dctrl-total50
#SBATCH --output=slurm_logs/6dctrl_total50_%j.out
#SBATCH --error=slurm_logs/6dctrl_total50_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── 6Dctrl final curriculum ablation ─────────────────────────────────────────
# Same env + estimator as the other 6Dctrl runs (R1 architecture, 6D wrench,
# TCN pre, no rec, pose commands). Differs only in the curriculum gate:
#   - force_gate_mode = "total" (default, same as R1/R2/R3)
#   - force_activation_reward_threshold = 50.0
# 6Dctrl total reward plateaus at ~49 while locomotion + pose tracking settle;
# this threshold fires forces just after that settles.

module load eth_proxy

TASK="Go2-Ablation-6Dctrl-Total50-v0"

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
