#!/bin/bash
#SBATCH --job-name=p23-p35
#SBATCH --output=slurm_logs/p23_p35_%a_%j.out
#SBATCH --error=slurm_logs/p23_p35_%a_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-19

# ── P23-P35 Ablation Study: 19 runs ──────────────────────────────────────────
# P23: PAINT trap h4          P24: PAINT trap h8 (baseline)  P25: PAINT trap h10
# P26: PAINT trap h8 no-rec   P27: PAINT trap h8 zero20%     P28: constant h8
# P29: cyclic trap h8         P30: constant h4               P31: constant h8b
# P32a: comp B30 s025         P32b: comp B30 s05             P32c: comp B40 s025
# P32d: comp B40 s05          P33: est-acc w50               P34: comp+acc B30
# P35: est-acc+TCN
#
# All: 30N force, 10Nm torque, fz_scale=0.8, 4D estimator

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1)  TASK="Go2-Ablation-P23-v0" ;;
    2)  TASK="Go2-Ablation-P24-v0" ;;
    3)  TASK="Go2-Ablation-P25-v0" ;;
    4)  TASK="Go2-Ablation-P26-v0" ;;
    5)  TASK="Go2-Ablation-P27-v0" ;;
    6)  TASK="Go2-Ablation-P28-v0" ;;
    7)  TASK="Go2-Ablation-P29-v0" ;;
    8)  TASK="Go2-Ablation-P30-v0" ;;
    9)  TASK="Go2-Ablation-P31-v0" ;;
    10) TASK="Go2-Ablation-P32a-v0" ;;
    11) TASK="Go2-Ablation-P32b-v0" ;;
    12) TASK="Go2-Ablation-P32c-v0" ;;
    13) TASK="Go2-Ablation-P32d-v0" ;;
    14) TASK="Go2-Ablation-P33-v0" ;;
    15) TASK="Go2-Ablation-P34-v0" ;;
    16) TASK="Go2-Ablation-P35-v0" ;;
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
