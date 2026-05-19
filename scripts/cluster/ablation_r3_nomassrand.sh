#!/bin/bash
#SBATCH --job-name=r3-nomassrand
#SBATCH --output=slurm_logs/r3_nomassrand_%j.out
#SBATCH --error=slurm_logs/r3_nomassrand_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# ── R3: R1 variant with fixed base mass (no mass randomization) ──────────────
# 6D wrench, persistent wrench, h=30, bigger net, TCN pre, no rec, no est-acc
# Same as R1, but add_base_mass event uses mass_distribution_params=(0,0).

module load eth_proxy

TASK="Go2-Ablation-R3-v0"

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
