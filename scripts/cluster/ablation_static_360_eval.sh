#!/bin/bash
#SBATCH --job-name=abl-eval360
#SBATCH --output=slurm_logs/ablation_eval360_%a_%j.out
#SBATCH --error=slurm_logs/ablation_eval360_%a_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-18

# ── Static 360 eval for batch 1 ablations ──────────────────────────────────
# 9 ablation tasks × 2 (nonlinear + linear) = 18 jobs
#
# Odd IDs  = nonlinear (no modulation)
# Even IDs = linear (with modulation)

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1)  TASK="Go2-Ablation-A1-v0"; LINEAR="" ;;
    2)  TASK="Go2-Ablation-A1-v0"; LINEAR="--linear_modulation" ;;
    3)  TASK="Go2-Ablation-A2-v0"; LINEAR="" ;;
    4)  TASK="Go2-Ablation-A2-v0"; LINEAR="--linear_modulation" ;;
    5)  TASK="Go2-Ablation-B1-v0"; LINEAR="" ;;
    6)  TASK="Go2-Ablation-B1-v0"; LINEAR="--linear_modulation" ;;
    7)  TASK="Go2-Ablation-B2-v0"; LINEAR="" ;;
    8)  TASK="Go2-Ablation-B2-v0"; LINEAR="--linear_modulation" ;;
    9)  TASK="Go2-Ablation-B3-v0"; LINEAR="" ;;
    10) TASK="Go2-Ablation-B3-v0"; LINEAR="--linear_modulation" ;;
    11) TASK="Go2-Ablation-C1-v0"; LINEAR="" ;;
    12) TASK="Go2-Ablation-C1-v0"; LINEAR="--linear_modulation" ;;
    13) TASK="Go2-Ablation-C2-v0"; LINEAR="" ;;
    14) TASK="Go2-Ablation-C2-v0"; LINEAR="--linear_modulation" ;;
    15) TASK="Go2-Ablation-E1-v0"; LINEAR="" ;;
    16) TASK="Go2-Ablation-E1-v0"; LINEAR="--linear_modulation" ;;
    17) TASK="Go2-Ablation-E2-v0"; LINEAR="" ;;
    18) TASK="Go2-Ablation-E2-v0"; LINEAR="--linear_modulation" ;;
    *) echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Task: $TASK"
echo "Linear modulation: ${LINEAR:-none}"
echo "========================================="

nvidia-smi

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab
python scripts/rsl_rl/eval/static_360_eval.py --task ${TASK} --num_trials 20 --force_hold_s 4.0 --warmup_s 3.0 --headless ${LINEAR}

echo "Job ${TASK} eval completed at $(date)"
