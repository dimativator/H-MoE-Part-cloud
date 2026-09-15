#!/usr/bin/env bash
set -u

root=$(cd "$(dirname "$0")" && pwd)
run_id=${RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}
results_root=${RESULTS_ROOT:-/workspace-SR006.nfs3/dimativator/megatron-muon-state-comm}
output_dir=${BENCHMARK_OUTPUT_DIR:-$results_root/$run_id}
log_dir=/home/jovyan/hmoe-cloud/logs
mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
mpi_size=${OMPI_COMM_WORLD_SIZE:-1}
mpi_local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
tp=${TP:-1}
pp=${PP:-2}
dp=${DP:-2}
transformer_impl=${TRANSFORMER_IMPL:-local}
expected_world_size=$((tp * pp * dp))
log=$log_dir/muon-state-comm-${run_id}-rank${mpi_rank}.log
mkdir -p "$log_dir" "$output_dir" "$root/runtime-tmp/triton-cache"

if (( mpi_size > 1 && mpi_size != expected_world_size )); then
    echo "mlsub MPI world size ${mpi_size} != TP*PP*DP=${expected_world_size}" >&2
    exit 8
fi
if (( mpi_size > 1 )); then
    export RANK=${RANK:-$mpi_rank}
    export WORLD_SIZE=${WORLD_SIZE:-$mpi_size}
    export LOCAL_RANK=${LOCAL_RANK:-$mpi_local_rank}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
    export MASTER_PORT=${MASTER_PORT:-29500}
fi

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
    export TRITON_CACHE_DIR="$root/runtime-tmp/triton-cache"
    python "$root/scripts/benchmark_muon_state_communication.py" \
        --output-dir "$output_dir" \
        --tensor-parallel-size "$tp" \
        --pipeline-parallel-size "$pp" \
        --data-parallel-size "$dp" \
        --transformer-impl "$transformer_impl" \
        "$@"
) >"$log" 2>&1
code=$?

echo "EXIT=$code"
echo "LOG=$log"
echo "OUTPUT_DIR=$output_dir"
tail -n 100 "$log"
if (( mpi_rank == 0 )) && [[ -f "$output_dir/results.csv" ]]; then
    echo "=== RESULTS CSV ==="
    cat "$output_dir/results.csv"
fi
exit "$code"
