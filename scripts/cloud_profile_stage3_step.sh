#!/usr/bin/env bash
set -euo pipefail

task_root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONUNBUFFERED=1
export PYTHONPATH="$task_root/third_party/Megatron-LM:$task_root/third_party/emerging-optimizers:$task_root"
export STAGE4_FP8_BATCHED=1
export BENCHMARK_OUTPUT_DIR=${BENCHMARK_OUTPUT_DIR:-/home/jovyan/hmoe-cloud/step-profile-20261013}
if [[ ${MLSUB_IMAGE:-} == torch28 ]]; then
    unset PYTHONNOUSERSITE
    task_nvidia_libs=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
        -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
    export LD_LIBRARY_PATH=${task_nvidia_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
    export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
    export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
else
    export PYTHONNOUSERSITE=1
fi
cd "$task_root"
exec python scripts/profile_stage3_training_step.py
