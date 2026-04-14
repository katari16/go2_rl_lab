#!/bin/bash
#SBATCH --job-name=est-ablation-s1-s9
#SBATCH --output=slurm_logs/ablation_s1_s9_%a_%j.out
#SBATCH --error=slurm_logs/ablation_s1_s9_%a_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-9

# ── Force Estimator Ablation: S1-S9 (frozen-policy) ────────────────────
# Base: Go2-LowLevel-NoEst-v0 (57-dim policy, 67-dim critic), actor frozen.
# Runner: FrozenPolicyEstimatorRunner — skips PPO, trains estimator only.
# Forces: 50N max, 10% force-free envs.
#
# S1: 2D (Fx, Fy),           h=30, default network
# S2: 3D (Fxyz),             h=30, default network
# S3: xy_yaw (Fx, Fy, τyaw), h=30, big network
# S4: 4D (Fxyz + τyaw),      h=30, default network, yaw_loss=3
# S5: 6D wrench,             h=30, big, equal loss weights
# S6: 6D wrench,             h=30, big, yaw_loss=3 + torque_angle_loss=3
# S7: 6D wrench,             h=30, big, linear temporal decay
# S8: 6D wrench,             h=30, big, TCN preprocessor
# S9: 6D wrench,             h=30, big, TCN replacement

module load eth_proxy

# Override with: sbatch --export=LOCO_CKPT=/path/to/model.pt ablation_S1_S9_sweep.sh
LOCO_CKPT="${LOCO_CKPT:-/cluster/home/habaumann/go2_rl_lab/logs/rsl_rl/go2_lowlevel_no_est/2026-04-13_23-59-10/model_11000.pt}"

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-S1-v0" ;;
    2) TASK="Go2-Ablation-S2-v0" ;;
    3) TASK="Go2-Ablation-S3-v0" ;;
    4) TASK="Go2-Ablation-S4-v0" ;;
    5) TASK="Go2-Ablation-S5-v0" ;;
    6) TASK="Go2-Ablation-S6-v0" ;;
    7) TASK="Go2-Ablation-S7-v0" ;;
    8) TASK="Go2-Ablation-S8-v0" ;;
    9) TASK="Go2-Ablation-S9-v0" ;;
    *) echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Task: $TASK"
echo "Locomotion checkpoint: $LOCO_CKPT"
echo "========================================="

nvidia-smi

if [ ! -f "$LOCO_CKPT" ]; then
    echo "ERROR: locomotion checkpoint not found at $LOCO_CKPT"
    exit 1
fi

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab
python scripts/rsl_rl/train.py --task ${TASK} --num_envs 4096 --headless --locomotion_checkpoint ${LOCO_CKPT}

echo "Job ${TASK} completed at $(date)"
