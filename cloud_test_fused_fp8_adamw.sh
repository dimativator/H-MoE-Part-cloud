#!/usr/bin/env bash
set -eu

if [[ ${OMPI_COMM_WORLD_RANK:-0} != 0 ]]; then
    exit 0
fi

root=$(cd "$(dirname "$0")" && pwd)
if [[ ${MLSUB_IMAGE:-} == torch28 ]]; then
    unset PYTHONNOUSERSITE
    nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
        -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
    export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"

python -m pytest -q \
    "$root/tests/stage4/test_fp8_optimizer_states.py" \
    -k fused_fp8_adamw_matches_storage_only_reference
