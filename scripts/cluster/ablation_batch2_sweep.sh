#!/bin/bash
#SBATCH --job-name=est-ablation-batch2
#SBATCH --output=slurm_logs/ablation_batch2_%a_%j.out
#SBATCH --error=slurm_logs/ablation_batch2_%a_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-18

# ── Force Estimator Ablation: Batch 2 (H-series dimension sweep) ─────────
# 9 configs × 2 force levels (50N / 100N) = 18 runs
#
# H1:  2D (Fx,Fy), h=30, rec, default net
# H3a: 6D wrench, h=40, rec, bigger net
# H3b: 6D wrench, h=30, rec, bigger net
# H3c: 6D wrench, h=30, rec, default net
# H5:  4D (Fx,Fy,Fz,tau_yaw), h=30, rec+yaw_loss, default net
# H6:  4D (Fx,Fy,Fz,tau_yaw), h=30, rec+yaw_loss, bigger net
# H7:  4D + force est accuracy reward, h=30, rec+yaw_loss, default net
# H8:  3D + force est accuracy reward, h=30, rec, default net
# H9:  3D (Fx,Fy,Fz), h=30, rec, default net

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1)  TASK="Go2-Ablation-H1-50N-v0" ;;
    2)  TASK="Go2-Ablation-H1-100N-v0" ;;
    3)  TASK="Go2-Ablation-H3a-50N-v0" ;;
    4)  TASK="Go2-Ablation-H3a-100N-v0" ;;
    5)  TASK="Go2-Ablation-H3b-50N-v0" ;;
    6)  TASK="Go2-Ablation-H3b-100N-v0" ;;
    7)  TASK="Go2-Ablation-H3c-50N-v0" ;;
    8)  TASK="Go2-Ablation-H3c-100N-v0" ;;
    9)  TASK="Go2-Ablation-H5-50N-v0" ;;
    10) TASK="Go2-Ablation-H5-100N-v0" ;;
    11) TASK="Go2-Ablation-H6-50N-v0" ;;
    12) TASK="Go2-Ablation-H6-100N-v0" ;;
    13) TASK="Go2-Ablation-H7-50N-v0" ;;
    14) TASK="Go2-Ablation-H7-100N-v0" ;;
    15) TASK="Go2-Ablation-H8-50N-v0" ;;
    16) TASK="Go2-Ablation-H8-100N-v0" ;;
    17) TASK="Go2-Ablation-H9-50N-v0" ;;
    18) TASK="Go2-Ablation-H9-100N-v0" ;;
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
python scripts/rsl_rl/train.py --task ${TASK} --num_envs 4096 --headless

echo "Job ${TASK} completed at $(date)"
