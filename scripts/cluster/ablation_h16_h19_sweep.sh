#!/bin/bash
#SBATCH --job-name=est-ablation-h16-h19
#SBATCH --output=slurm_logs/ablation_h16_h19_%a_%j.out
#SBATCH --error=slurm_logs/ablation_h16_h19_%a_%j.err
#SBATCH --time=30:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-4

# ── Force Estimator Ablation: H16-H19 ──────────────────────────────────
# All 50N, 6D wrench env (except H19 which reads [Fx, Fy, τ_yaw])
# H16: equal loss weights (all 1.0) — sensitivity test
# H17: linear decay temporal weighting — bias toward recent obs
# H18: TCN preprocessor + detached recon loss — learned temporal features
# H19: 3D planar wrench [Fx, Fy, τ_yaw] — minimal set for planar compliance

module load eth_proxy

case $SLURM_ARRAY_TASK_ID in
    1) TASK="Go2-Ablation-H16-50N-v0" ;;
    2) TASK="Go2-Ablation-H17-50N-v0" ;;
    3) TASK="Go2-Ablation-H18-50N-v0" ;;
    4) TASK="Go2-Ablation-H19-50N-v0" ;;
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
