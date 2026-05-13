#!/bin/bash
#SBATCH --job-name=vlm-sc20-eval
#SBATCH --partition=3090_risk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02-00:00:00
#SBATCH --output=/mnt/fast/nobackup/scratch4weeks/vt00262/projects/IFSL-VL/_experiments/slurm_logs/%x_%j.out

# Single-run VLM eval on ScanNet200: ScanNet-20 as base; novel = CLASS_LABELS_BASE_NOVEL
# minus those bases (non-long-tail set; see CLASS_LABELS_SC20_BN_COMPLEMENT_NOVEL).
# Log: ${SAVE_ROOT}/test.log
#
# Usage:
#   sbatch scripts/eval_vlm_novel_scannet200.sh
#   bash scripts/eval_vlm_novel_scannet200.sh
#
# Optional env:
#   WORK_DIR=...        # default: repo root inferred from this script
#   CONFIG=configs/scannet/vlm-eval-scannet200-sc20base-180novel.py
#   SAVE_ROOT=_experiments/vlm_zero_shot_scannet200
#   NUM_GPUS=1          # or SLURM_GPUS_ON_NODE when under Slurm
#   VLM_3D_WEIGHT=...   # overrides config vlm_3d_weight

set -euo pipefail

RUN_ID="${SLURM_JOB_ID:-DEBUG}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}}"
cd "${WORK_DIR}"

export PYTHONPATH="${WORK_DIR}:${WORK_DIR}/pointcept/models/PLA${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-configs/scannet/vlm-eval-scannet200-sc20base-180novel.py}"
SAVE_ROOT="${SAVE_ROOT:-_experiments/vlm_zero_shot_scannet200}"
VLM_3D_WEIGHT="${VLM_3D_WEIGHT:-}"

mkdir -p "${SAVE_ROOT}"

echo "Work dir: ${WORK_DIR}"
echo "Config:   ${CONFIG}"
echo "Save:     ${SAVE_ROOT}"
echo "GPUs:     ${NUM_GPUS}"
if [[ -n "${VLM_3D_WEIGHT}" ]]; then
  echo "VLM ckpt: ${VLM_3D_WEIGHT} (override)"
  OPT=(save_path="${SAVE_ROOT}" vlm_3d_weight="${VLM_3D_WEIGHT}")
else
  echo "VLM ckpt: (from config file)"
  OPT=(save_path="${SAVE_ROOT}")
fi
echo

aprun \
  --env "SLURM_JOB_ID=${RUN_ID}" \
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python tools/eval_vlm_novel.py \
    --config-file "${CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --split sc20_180 \
    --options "${OPT[@]}"

echo "Wrote: ${SAVE_ROOT}/test.log"
