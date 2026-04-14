#!/bin/bash
#SBATCH --job-name=ifsl-mbuffer
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02-00:00:00
#SBATCH --output=/mnt/fast/nobackup/scratch4weeks/vt00262/projects/IFSL-VL/_experiments/slurm_logs/%x_%j.out

# Srun command for debugging
# srun -p 3090_risk --gres=gpu:2 --ntasks=2 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash

# Exit on error
set -e

RUN_ID=${SLURM_JOB_ID:-"DEBUG"}
NUM_GPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

# -------------------------------
# Paths
# -------------------------------
# HOST=$(hostname)
# WORK_DIR="/mnt/fast/nobackup/scratch4weeks/vt00262/projects/CLIMB3D++"

# -------------------------------
# Experiment Config
# -------------------------------
EXP_DIR="_experiments/scannet"
# EXP_NAME="${RUN_ID}_gfs_novel_registration_wit_mbuffer"
EXP_NAME="DEBUG_gfs_novel_registration_wit_mbuffer"

EXP_PATH=${EXP_DIR}/${EXP_NAME}
# while [ -d "${EXP_PATH}" ]; do
#   RAND_ID=$(tr -dc 'a-z0-9' </dev/urandom | head -c 4)
#   EXP_NAME="${EXP_NAME}_${RAND_ID}"
#   EXP_PATH=${EXP_DIR}/${EXP_NAME}
# done

# -------------------------------
# Run
# -------------------------------
# HOST=$(hostname)
# if [[ "$HOST" != "ulws102.surrey.ac.uk" ]]; then
#     cd "$WORK_DIR"
# fi

aprun \
  --env SLURM_JOB_ID=$RUN_ID \
  --env OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python tools/train.py \
    --config-file configs/scannet/semseg-pt-v3m1-0-gfsregistrain_k5.py \
    --num-gpus $NUM_GPUS \
    --options save_path=${EXP_PATH}
