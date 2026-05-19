#!/bin/bash
#SBATCH --job-name=r8-buckets
#SBATCH --output=slurm_logs/r8_buckets_%j.out
#SBATCH --error=slurm_logs/r8_buckets_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── R8: R1 with bucketed force curriculum post-gate ──────────────────────────
# Same R1 env and estimator. Reward gate fires as usual (~iter 2100); then
# bucket 1 = (0,10)N × 1000 iters, bucket 2 = (0,20)N × 1000 iters,
# bucket 3 = (0,30)N for the remainder. Torque max scales proportionally.

module load eth_proxy

TASK="Go2-Ablation-R8-v0"

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
