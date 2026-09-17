#!/usr/bin/env bash
set -u

if [[ ${OMPI_COMM_WORLD_RANK:-0} != 0 ]]; then
    exit 0
fi

root=$(cd "$(dirname "$0")" && pwd)
output_dir=${BENCHMARK_OUTPUT_DIR:-/home/jovyan/hmoe-cloud/step-time}
log_dir=/home/jovyan/hmoe-cloud/logs
log=$log_dir/training-step-$(date +%F_%H%M%S).log
mkdir -p "$log_dir" "$output_dir"

(
    set -eu
    if [[ ${MLSUB_IMAGE:-} == torch28 ]]; then
        unset PYTHONNOUSERSITE
        nvidia_lib_path=$(find /home/user/conda/lib/python3.12/site-packages/nvidia \
            -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd: -)
        export LD_LIBRARY_PATH=${nvidia_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        export CUDNN_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cudnn
        export CURAND_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/curand
        export NVRTC_HOME=/home/user/conda/lib/python3.12/site-packages/nvidia/cuda_nvrtc
    else
        export PYTHONNOUSERSITE=1
    fi
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="$root/third_party/Megatron-LM:$root/third_party/emerging-optimizers:$root"
    export STAGE4_FP8_BATCHED=1
    python "$root/scripts/benchmark_training_step.py" --output-dir "$output_dir" "$@"
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
echo "OUTPUT_DIR=$output_dir"
tail -n 80 "$log"
if [[ -f "$output_dir/results.csv" ]]; then
    echo "=== RESULTS CSV ==="
    cat "$output_dir/results.csv"
fi
if [[ -f "$output_dir/results.json" ]]; then
    python - "$output_dir/results.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print("=== BENCHMARK METADATA ===")
print(json.dumps(payload["metadata"], sort_keys=True))
PY
fi
exit 0
