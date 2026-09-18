#!/usr/bin/env bash
set -euo pipefail

if [[ ${OMPI_COMM_WORLD_RANK:-0} != 0 ]]; then
    exit 0
fi

root=$(cd "$(dirname "$0")" && pwd)
group=${MUON_FP8_SUITE_GROUP:?MUON_FP8_SUITE_GROUP must be A, B, or C}
output_root=${BENCHMARK_OUTPUT_ROOT:-/home/jovyan/hmoe-cloud/bf16-muon-fp8-states-step-50}

run_model() {
    local model=$1
    local micro_batch=$2
    local output_dir="$output_root/$model"

    BENCHMARK_OUTPUT_DIR="$output_dir" "$root/cloud_benchmark_training_step.sh" \
        --models "$model" \
        --precisions bf16 \
        --optimizers muon \
        --batches 1,2,4,8,16,32 \
        --micro-batch-size "$micro_batch" \
        --warmup-steps 10 \
        --measure-steps 50 \
        --muon-state-precision fp8
}

case "$group" in
    A)
        run_model 257m 32
        run_model 3b 1
        ;;
    B)
        run_model 500m 16
        run_model 5b 1
        ;;
    C)
        run_model 1b 8
        ;;
    *)
        echo "unsupported MUON_FP8_SUITE_GROUP=$group" >&2
        exit 2
        ;;
esac
