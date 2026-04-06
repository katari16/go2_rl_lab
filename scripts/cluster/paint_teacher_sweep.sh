#!/bin/bash
#SBATCH --job-name=paint-teacher
#SBATCH --output=slurm_logs/paint_teacher_%a_%j.out
#SBATCH --error=slurm_logs/paint_teacher_%a_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-2

# ── PAINT Stage 1: Teacher training (Group D) ───────────────────────────────
# D1-Teacher: 3D force, compliance reward env (same env as E1)
# D2-Teacher: 4D wrench, compliance reward env (same env as E2)
#
# These use CompliantOnPolicyRunner (standard 3-phase training).
# After training, use the checkpoint as teacher for Stage 2 (paint_student_sweep.sh).

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-D1-Teacher-v0" ;;
    2) TASK="Go2-Ablation-D2-Teacher-v0" ;;
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
