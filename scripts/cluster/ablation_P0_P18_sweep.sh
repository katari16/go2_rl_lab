#!/bin/bash
#SBATCH --job-name=ablation-p0-p18
#SBATCH --output=slurm_logs/ablation_p_%a_%j.out
#SBATCH --error=slurm_logs/ablation_p_%a_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=0-18

# ── PAINT Force Profile Ablation: P0-P18 ───────────────────────────────────
# Base: CompliantOnPolicyRunner + PAINT trapezoid force profile (10/80/10 ramp).
# Forces: 20N max (XYZ), torque 5Nm, stratified 4-bucket, zero_prob=0.05.
#
# P0:  Baseline — h=30, 4D, default net, rec loss
# P1:  History h=10
# P2:  History h=20
# P3:  History h=30 (same as P0)
# P4:  History h=40
# P5:  Network half (enc=[64,32], f_head=[16,8])
# P6:  Network double (enc=[256,128], f_head=[64,32])
# P7:  Estimator accuracy reward w=50
# P8:  Compliance reward w=0.5
# P9:  Compliance reward w=1.0
# P10: Compliance reward w=5.0
# P11: No reconstruction loss
# P12: TCN encoder
# P13: Force dim 2D (fx, fy)
# P14: Force dim xy_yaw (fx, fy, τ_yaw)
# P15: Force dim 4D (baseline)
# P16: Force dim 6D (default net)
# P17: Force dim 6D (big net)
# P18: Payload with randomized mass 0-4kg per episode

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    0)  TASK="Go2-Ablation-P0-v0" ;;
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
