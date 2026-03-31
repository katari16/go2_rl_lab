#!/bin/bash
#SBATCH --job-name=go2-ft-v3
#SBATCH --output=slurm_logs/finetune_v3_retrain_%j.out
#SBATCH --error=slurm_logs/finetune_v3_retrain_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G

# V3 retrain from scratch: R2 + standing pose (rough terrain, no stairs)
# Low PD gains (Kp=8), 20k iterations

module load eth_proxy

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Task: Go2-Finetune-V3-v0 (retrain from scratch)"
echo "========================================="

nvidia-smi

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /cluster/home/habaumann/go2_rl_lab
python scripts/rsl_rl/train.py \
    --task Go2-Finetune-V3-v0 \
    --headless \
    --num_envs 4096

echo "V3 retrain completed at $(date)"
