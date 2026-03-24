#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------
# Multi-architecture CUDA build (portable across GPUs)
# ---------------------------------------------------------
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
echo "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# ---------------------------------------------------------
# Number of parallel workers
# ---------------------------------------------------------
NUM_WORKERS=${NUM_WORKERS:-$(nproc)}
export MAX_JOBS=${NUM_WORKERS}
export CUDA_NVCC_THREADS=${MAX_JOBS}

echo "Using ${MAX_JOBS} parallel build workers (NVCC + PyTorch)"

# Optional (for softgroup / some CUDA builds)
export CPLUS_INCLUDE_PATH=/opt/micromamba/include:${CPLUS_INCLUDE_PATH:-}

# ---------------------------------------------------------
# Helper: clean build artifacts
# ---------------------------------------------------------
clean_build () {
    python setup.py clean >/dev/null 2>&1 || true
    rm -rf build dist *.egg-info
}

# # ---------------------------------------------------------
# # Build libs/pointops
# # ---------------------------------------------------------
# echo "Building pointops..."
cd libs/pointops
# clean_build
# python setup.py bdist_wheel

# # ---------------------------------------------------------
# # Build libs/pointops2
# # ---------------------------------------------------------
# echo "Building pointops2..."
cd ../pointops2
# clean_build
# python setup.py bdist_wheel

# ---------------------------------------------------------
# Build libs/pointgroup_ops (optional)
# ---------------------------------------------------------
# echo "Building pointgroup_ops..."
# cd ../pointgroup_ops
# clean_build
# python setup.py bdist_wheel

# ---------------------------------------------------------
# Build softgroup_ops (FIXED PATH)
# ---------------------------------------------------------
# echo "Building softgroup_ops..."
# cd ../../pcseg/external_libs/softgroup_ops
# clean_build
# python setup.py bdist_wheel

# ---------------------------------------------------------
# Build PLA model
# ---------------------------------------------------------
echo "Building PLA..."
cd ../../pointcept/models/PLA
clean_build
python setup.py bdist_wheel

# ---------------------------------------------------------
# Build FlashAttention (optional)
# ---------------------------------------------------------
# git clone --recursive https://github.com/Dao-AILab/flash-attention.git && \
# cd flash-attention && \
# git checkout ef736fe03d42c0d5f0844486e57ebf6c6dd12d0b && \
# git submodule update --init --recursive

# FlashAttention not reliable on < sm_80
# echo "Building flash-attention..."
# cd ../../../libs/flash-attention
# export FLASH_ATTN_CUDA_ARCHS="70;75;80;86;89;90"
# clean_build
# python setup.py bdist_wheel

# ---------------------------------------------------------
# Done
# ---------------------------------------------------------
echo "✅ All wheels built successfully."


# apptainer shell --writable --nv --cleanenv \
#     --env PATH="/opt/micromamba/bin:/opt/micromamba:$PATH" \
#     --bind /vol/research/Vishal_Thengane/projects/IFSL-VL:/vol/research/Vishal_Thengane/projects/IFSL-VL \
#     /vol/research/Vishal_Thengane/containers/ifsl-vl/

