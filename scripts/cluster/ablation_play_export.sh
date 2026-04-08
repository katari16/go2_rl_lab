#!/bin/bash
#SBATCH --job-name=abl-export
#SBATCH --output=slurm_logs/ablation_export_%a_%j.out
#SBATCH --error=slurm_logs/ablation_export_%a_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-9

# ── Export JIT models for ablation batch 1 (9 runs) ─────────────────────────
# A1: 3D, h=10, rec=yes, default net
# A2: 3D, h=40, rec=yes, default net
# B1: 6D, h=10, rec=yes, default net
# B2: 6D, h=40, rec=yes, default net
# B3: 6D, h=40, rec=yes, bigger net
# C1: 3D, h=10, rec=no,  default net
# C2: 3D, h=40, rec=no,  default net
# E1: 3D, h=20, rec=yes, compliance force reward
# E2: 4D, h=20, rec=yes, compliance force+torque reward

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-A1-v0" ;;
    2) TASK="Go2-Ablation-A2-v0" ;;
    3) TASK="Go2-Ablation-B1-v0" ;;
    4) TASK="Go2-Ablation-B2-v0" ;;
    5) TASK="Go2-Ablation-B3-v0" ;;
    6) TASK="Go2-Ablation-C1-v0" ;;
    7) TASK="Go2-Ablation-C2-v0" ;;
    8) TASK="Go2-Ablation-E1-v0" ;;
    9) TASK="Go2-Ablation-E2-v0" ;;
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
python scripts/rsl_rl/play_export.py --task ${TASK} --num_envs 1 --headless

echo "Job ${TASK} export completed at $(date)"
