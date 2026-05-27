#!/bin/bash
#SBATCH --job-name=gfs-mbuffer
#SBATCH --partition=3090_risk
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
# Local runs default RUN_ID=DEBUG → enable fast debug limits unless overridden
if [[ "$RUN_ID" == "DEBUG" ]]; then
  DEBUG=${DEBUG:-1}
else
  DEBUG=${DEBUG:-0}
fi

# -------------------------------
# Paths
# -------------------------------
# WORK_DIR="/mnt/fast/nobackup/scratch4weeks/vt00262/projects/IFSL-VL"
# cd $WORK_DIR

# -------------------------------
# Experiment Config
# -------------------------------
# DEBUG=1 (auto when RUN_ID=DEBUG): 10 train, 5 eval, 1 regis, 2 epochs

EXP_DIR="_experiments/scannet"
if [[ "$DEBUG" == "1" ]]; then
  EXP_NAME="${RUN_ID}_gfs_debug"
else
  EXP_NAME="${RUN_ID}_gfs_novel_registration_wit_mbuffer_v2"
fi
EXP_PATH=${EXP_DIR}/${EXP_NAME}

EXTRA_OPTS=()
if [[ "$DEBUG" == "1" ]]; then
  EXTRA_OPTS+=(
    save_path=${EXP_PATH}
    max_train_samples=10
    max_eval_samples=5
    data.train.memory_ratio=0.005
    epoch=2
    eval_epoch=2
  )
else
  EXTRA_OPTS+=(save_path=${EXP_PATH})
fi

# -------------------------------
# Run
# -------------------------------

aprun \
  --env SLURM_JOB_ID=$RUN_ID \
  --env OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python tools/train.py \
    --config-file configs/scannet/semseg-pt-v3m1-0-gfsregistrain_k5.py \
    --num-gpus $NUM_GPUS \
    --options "${EXTRA_OPTS[@]}"
