#!/bin/bash
#SBATCH --job-name=6dctrl
#SBATCH --output=slurm_logs/6dctrl_%j.out
#SBATCH --error=slurm_logs/6dctrl_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── 6Dctrl: R1 + commanded roll/pitch/height ─────────────────────────────────
# UniformVelocityPoseCommand with absolute height in [0.24, 0.38] m,
# roll ±0.25 rad, pitch ±0.30 rad. 20% nominal probability per pose channel.
# Same estimator config as R1 (h=30, 6D big TCN no-rec, no est-acc reward).

module load eth_proxy

TASK="Go2-Ablation-6Dctrl-v0"

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
