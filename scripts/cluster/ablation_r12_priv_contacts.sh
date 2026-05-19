#!/bin/bash
#SBATCH --job-name=r12-priv-contacts
#SBATCH --output=slurm_logs/r12_priv_contacts_%j.out
#SBATCH --error=slurm_logs/r12_priv_contacts_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── R12: R1 + privileged foot contact force norms only (peel-back) ───────────
# Same as R9 but only the 4 foot contact force norms (FL/FR/RL/RR, scaled 0.01)
# are added to the estimator input. Isolates how much of the privileged-info
# gain comes from knowing the ground reaction forces (the J_c^T F_c term).

module load eth_proxy

TASK="Go2-Ablation-R12-v0"

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
