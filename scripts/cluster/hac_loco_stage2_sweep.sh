#!/bin/bash
#SBATCH --job-name=hac-s2-sweep
#SBATCH --output=slurm_logs/hac_s2_R%a_%j.out
#SBATCH --error=slurm_logs/hac_s2_R%a_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=10G
#SBATCH --array=1-8

# ── HAC-LOCO Stage 2 Sweep: 8 variations ─────────────────────────────────
# R1: penalty,  no gravity corr,  tracking reward
# R2: positive, no gravity corr,  tracking reward
# R3: penalty,  gravity corr,     tracking reward
# R4: positive, gravity corr,     tracking reward
# R5: penalty,  no gravity corr,  NO tracking reward
# R6: positive, no gravity corr,  NO tracking reward
# R7: penalty,  gravity corr,     NO tracking reward
# R8: positive, gravity corr,     NO tracking reward
#
# All: V3 low-level frozen, rough terrain, w_comp=2.0, clip=3.0, 5k iter

module load eth_proxy

echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Task: Go2-HighLevel-NonLinear-R${SLURM_ARRAY_TASK_ID}-v0"
echo "========================================="

nvidia-smi

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate env_isaaclab

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

STAGE1_CKPT="/cluster/home/habaumann/go2_rl_lab/logs/rsl_rl/go2_finetune_v3/2026-03-10_13-04-30/model_12500.pt"

cd /cluster/home/habaumann/go2_rl_lab
python scripts/rsl_rl/train.py \
    --task Go2-HighLevel-NonLinear-R${SLURM_ARRAY_TASK_ID}-v0 \
    --num_envs 4096 \
    --stage1_checkpoint ${STAGE1_CKPT} \
    --headless

echo "Job R${SLURM_ARRAY_TASK_ID} completed at $(date)"
