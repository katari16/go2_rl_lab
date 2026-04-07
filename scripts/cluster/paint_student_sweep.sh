#!/bin/bash
#SBATCH --job-name=paint-student
#SBATCH --output=slurm_logs/paint_student_%a_%j.out
#SBATCH --error=slurm_logs/paint_student_%a_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-2

# ── PAINT Stage 2: Student distillation (Group D) ───────────────────────────
# D1-Student: PAINT distillation, 3D force, compliance reward env
# D2-Student: PAINT distillation, 4D wrench, compliance reward env
#
# Teacher checkpoints from Stage 1 (paint_teacher_sweep.sh).
#
# Usage:
#   sbatch scripts/cluster/paint_student_sweep.sh

module load eth_proxy

TEACHER_D1="/cluster/home/habaumann/go2_rl_lab/logs/rsl_rl/ablation_D1_teacher/2026-04-06_23-52-15/model_14000.pt"
TEACHER_D2="/cluster/home/habaumann/go2_rl_lab/logs/rsl_rl/ablation_D2_teacher/2026-04-06_23-52-15/model_14000.pt"

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-D1-v0"; CKPT="$TEACHER_D1" ;;
    2) TASK="Go2-Ablation-D2-v0"; CKPT="$TEACHER_D2" ;;
    *) echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Task: $TASK"
echo "Teacher checkpoint: $CKPT"
echo "========================================="

nvidia-smi

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab
python scripts/rsl_rl/train.py \
    --task ${TASK} \
    --num_envs 4096 \
    --headless \
    --resume \
    --checkpoint ${CKPT}

echo "Job ${TASK} completed at $(date)"
