#!/bin/bash
#SBATCH --job-name=6dctrl-stand
#SBATCH --output=slurm_logs/6dctrl_stand_%a_%j.out
#SBATCH --error=slurm_logs/6dctrl_stand_%a_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-3

# ── 6Dctrl stand-still ablations ────────────────────────────────────────────
# Base = Ablation6DctrlTotal50Cfg (R1 estimator arch, force_gate total @ 50,
# 6D wrench, pose commands). Each variant flips one knob to fix the observed
# "robot never stops under zero command" behavior.
#   1) StandEnv10 — rel_standing_envs 0.02 -> 0.10 (more null-cmd envs)
#   2) StandW2    — standing_pose weight -0.5 -> -2.0 (stronger null penalty)
#   3) StandBoth  — both combined

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-6Dctrl-StandEnv10-v0" ;;
    2) TASK="Go2-Ablation-6Dctrl-StandW2-v0" ;;
    3) TASK="Go2-Ablation-6Dctrl-StandBoth-v0" ;;
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
