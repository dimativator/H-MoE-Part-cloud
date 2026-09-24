#!/usr/bin/env bash
set -euo pipefail

if [[ ${OMPI_COMM_WORLD_RANK:-0} != 0 ]]; then
    exit 0
fi

root=$(cd "$(dirname "$0")" && pwd)
output=${DIAG_OUTPUT:-/home/jovyan/hmoe-cloud/adam-fp8-state-diagnosis.log}

if [[ ${MLSUB_IMAGE:-} == torch28 ]]; then
    unset PYTHONNOUSERSITE
    nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
        -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
    export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
    export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
    export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
mkdir -p "$(dirname "$output")"
python "$root/scripts/diagnose_adam_fp8_states.py" 2>&1 | tee "$output"
