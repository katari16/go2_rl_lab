#!/bin/bash
#SBATCH --job-name=go2-finetune
#SBATCH --output=slurm_logs/finetune_j%a_%j.out
#SBATCH --error=slurm_logs/finetune_j%a_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-8

module load eth_proxy

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Task: Go2-Finetune-J${SLURM_ARRAY_TASK_ID}-v0"
echo "========================================="

nvidia-smi

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab
python scripts/rsl_rl/train.py \
    --task Go2-Finetune-J${SLURM_ARRAY_TASK_ID}-v0 \
    --max_iterations 25000 \
    --headless

echo "Job J${SLURM_ARRAY_TASK_ID} completed at $(date)"
