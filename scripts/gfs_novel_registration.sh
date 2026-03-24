#!/bin/bash
#SBATCH --job-name=ifsl-vl-reproduce
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02-00:00:00
#SBATCH --output=/mnt/fast/nobackup/scratch4weeks/vt00262/projects/IFSL-VL/_experiments/slurm_logs/%x_%j.out
#SBATCH --error=/mnt/fast/nobackup/scratch4weeks/vt00262/projects/IFSL-VL/_experiments/slurm_logs/%x_%j.err

# srun -p 3090 --gres=gpu:2 --ntasks=2 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash

# -----------------------------
# Environment
# -----------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS=${SLURM_GPUS_ON_NODE:-${CUDA_VISIBLE_DEVICES:+$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}}
NUM_GPUS=${NUM_GPUS:-1}

# -----------------------------
# Run (IMPORTANT)
# -----------------------------
aprun python tools/train.py \
  --config-file configs/scannet/semseg-pt-v3m1-0-gfsregistrain_k5.py \
  --num-gpus $NUM_GPUS \
  --options save_path=_experiments/scannet/few_shot_registration
