#!/usr/bin/env bash
set -euo pipefail

if [[ ${OMPI_COMM_WORLD_RANK:-0} != 0 ]]; then
    exit 0
fi

root=$(cd "$(dirname "$0")" && pwd)
group=${OPTIMIZER_SUITE_GROUP:?OPTIMIZER_SUITE_GROUP must be A, B, or C}
output_root=${BENCHMARK_OUTPUT_ROOT:-/home/jovyan/hmoe-cloud/bf16-optimizer-adaptive-microbatch-step-50}
optimizers=adam,adam_fp8_states,muon,muon_fp8_states,soap,ademamix,galore,frugal,frugal_muon_muon,slim_adam,apollo

run_models() {
    local models=$1
    local output_dir="$output_root/$models"

    BENCHMARK_OUTPUT_DIR="$output_dir" "$root/cloud_benchmark_training_step.sh" \
        --models "$models" \
        --precisions bf16 \
        --optimizers "$optimizers" \
        --batches 1,2,4,8,16,32 \
        --micro-batch-size 32 \
        --adaptive-micro-batch \
        --warmup-steps 10 \
        --measure-steps 50
}

case "$group" in
    A)
        run_models 257m,500m,1b
        ;;
    B)
        run_models 3b
        ;;
    C)
        run_models 5b
        ;;
    *)
        echo "unsupported OPTIMIZER_SUITE_GROUP=$group" >&2
        exit 2
        ;;
esac
