#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
mpi_size=${OMPI_COMM_WORLD_SIZE:-1}
if (( mpi_size != 8 )); then
    echo "DP8 scaling requires exactly 8 MPI ranks, got $mpi_size" >&2
    exit 8
fi

run_prefix=${RUN_ID_PREFIX:-dp8-scaling-$(date +%Y%m%d-%H%M%S)}
methods=muon_bf16_states,muon_fp8_states,frugal_muon_muon_bf16_states,frugal_muon_muon_fp8_states
common=(
    --methods "$methods"
    --sequence-length 1024
    --micro-batch-size 1
    --global-batch-size 8
    --warmup-steps 10
    --measure-steps 50
    --repeats 3
    --fp8-bucket-bytes 67108864
    --fused-fp8-ns-input
)

for model in 1b 2b 3b 3.5b 4b 4.5b 4.9b; do
    echo "=== START model=$model ==="
    RUN_ID="${run_prefix}-p1dp8-${model}" TP=1 PP=1 DP=8 \
        "$root/cloud_benchmark_muon_state_communication.sh" \
        --models "$model" "${common[@]}"
    echo "=== END model=$model ==="
done
