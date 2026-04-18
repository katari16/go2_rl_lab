#!/bin/bash
#SBATCH --job-name=p-series
#SBATCH --output=slurm_logs/p_series_%a_%j.out
#SBATCH --error=slurm_logs/p_series_%a_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-22

# ── P-series Ablation Study: 22 runs ──────────────────────────────────────
# P1:  h10           P2:  h20           P3:  h30 (baseline)
# P4:  h40           P5:  half net      P6:  double net
# P7:  est rew w50   P8:  comp w0.5     P9:  comp w1.0
# P10: comp w5.0     P11: no rec loss   P12: TCN encoder
# P13: 2D            P14: xy_yaw        P15: 4D (=P3)
# P16: 6D default    P17: 6D big        P18: payload
# P19: torque 5Nm    P20: default PD    P21: trapezoid
# P22: stratified buckets
#
# All: 30N, 10Nm (except P19=5Nm), constant wrench, fz_scale=0.8

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1)  TASK="Go2-Ablation-P1-v0" ;;
    2)  TASK="Go2-Ablation-P2-v0" ;;
    3)  TASK="Go2-Ablation-P3-v0" ;;
    4)  TASK="Go2-Ablation-P4-v0" ;;
    5)  TASK="Go2-Ablation-P5-v0" ;;
    6)  TASK="Go2-Ablation-P6-v0" ;;
    7)  TASK="Go2-Ablation-P7-v0" ;;
    8)  TASK="Go2-Ablation-P8-v0" ;;
    9)  TASK="Go2-Ablation-P9-v0" ;;
    10) TASK="Go2-Ablation-P10-v0" ;;
    11) TASK="Go2-Ablation-P11-v0" ;;
    12) TASK="Go2-Ablation-P12-v0" ;;
    13) TASK="Go2-Ablation-P13-v0" ;;
    14) TASK="Go2-Ablation-P14-v0" ;;
    15) TASK="Go2-Ablation-P15-v0" ;;
    16) TASK="Go2-Ablation-P16-v0" ;;
    17) TASK="Go2-Ablation-P17-v0" ;;
    18) TASK="Go2-Ablation-P18-v0" ;;
    19) TASK="Go2-Ablation-P19-v0" ;;
    20) TASK="Go2-Ablation-P20-v0" ;;
    21) TASK="Go2-Ablation-P21-v0" ;;
    22) TASK="Go2-Ablation-P22-v0" ;;
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
