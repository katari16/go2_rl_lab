#!/bin/bash
#SBATCH --job-name=est-ablation-B4B5
#SBATCH --output=slurm_logs/ablation_B4B5_%a_%j.out
#SBATCH --error=slurm_logs/ablation_B4B5_%a_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-2

# ── Force Estimator Ablation: B4 & B5 (torque losses) ──────────────────────
# B4: 6D, h=40, rec=yes, bigger net, torque_angle_loss + yaw_loss
# B5: 6D, h=60, rec=yes, bigger net, torque_angle_loss + yaw_loss

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-B4-v0" ;;
    2) TASK="Go2-Ablation-B5-v0" ;;
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
