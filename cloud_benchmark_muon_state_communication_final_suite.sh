#!/usr/bin/env bash
set -eu

root=$(cd "$(dirname "$0")" && pwd)
mpi_size=${OMPI_COMM_WORLD_SIZE:-1}
run_prefix=${RUN_ID_PREFIX:-wire2-final-$(date +%Y%m%d-%H%M%S)}
methods=muon_bf16_states,muon_fp8_states,frugal_muon_muon_bf16_states,frugal_muon_muon_fp8_states
common=(
    --methods "$methods"
    --sequence-length 1024
    --micro-batch-size 1
    --warmup-steps 10
    --measure-steps 50
    --repeats 3
    --fp8-bucket-bytes 67108864
    --fused-fp8-ns-input
)

if (( mpi_size == 4 )); then
    RUN_ID="${run_prefix}-p1dp4" TP=1 PP=1 DP=4 \
        "$root/cloud_benchmark_muon_state_communication.sh" \
        --models 4.9b,5.0b-wide \
        --global-batch-size 4 \
        "${common[@]}"
elif (( mpi_size == 8 )); then
    RUN_ID="${run_prefix}-p1dp8" TP=1 PP=1 DP=8 \
        "$root/cloud_benchmark_muon_state_communication.sh" \
        --models 4.9b,5.0b-wide \
        --global-batch-size 8 \
        "${common[@]}"
    RUN_ID="${run_prefix}-pp2dp4" TP=1 PP=2 DP=4 \
        "$root/cloud_benchmark_muon_state_communication.sh" \
        --models 9.9b-pp2 \
        --global-batch-size 4 \
        "${common[@]}"
else
    echo "final suite requires exactly 4 or 8 MPI ranks, got ${mpi_size}" >&2
    exit 8
fi
