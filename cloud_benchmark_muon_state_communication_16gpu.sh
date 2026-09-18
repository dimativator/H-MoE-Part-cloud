#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
world_rank=${OMPI_COMM_WORLD_RANK:?16-GPU launch requires an MPI world rank}
local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:?16-GPU launch requires an MPI local rank}
world_size=${OMPI_COMM_WORLD_SIZE:?16-GPU launch requires an MPI world size}
local_size=${OMPI_COMM_WORLD_LOCAL_SIZE:?16-GPU launch requires an MPI local size}
if (( local_size < 1 || world_size % local_size != 0 )); then
    echo "invalid MPI layout: world_size=$world_size local_size=$local_size" >&2
    exit 8
fi
nnodes=$((world_size / local_size))
node_rank=$((world_rank / local_size))
leader=0
if (( local_rank == 0 )); then
    leader=1
fi
if [[ "$nnodes" != 2 ]]; then
    echo "expected two mlsub workers, got $nnodes" >&2
    exit 8
fi
if [[ "$world_size" != 16 || "$local_size" != 8 ]]; then
    echo "expected 2x8 MPI ranks, got world_size=$world_size local_size=$local_size" >&2
    exit 8
fi

if [[ ${MLSUB_LAYOUT_ONLY:-0} == 1 ]]; then
    echo "world_rank=$world_rank local_rank=$local_rank world_size=$world_size local_size=$local_size node_rank=$node_rank nnodes=$nnodes leader=$leader"
    exit 0
fi

run_prefix=${RUN_ID_PREFIX:-multinode16-$(date +%Y%m%d-%H%M%S)}
results_root=${RESULTS_ROOT:-/workspace-SR006.nfs3/dimativator/megatron-muon-state-comm}
run_root=$results_root/$run_prefix
log_dir=$run_root/node-logs
if (( leader )); then
    log=$log_dir/muon-state-comm-${run_prefix}-node${node_rank}.log
else
    log=$log_dir/muon-state-comm-${run_prefix}-mpi-rank${world_rank}.log
fi
mkdir -p "$log_dir" "$run_root/topology" "$root/runtime-tmp/triton-cache-node${node_rank}"
exec > >(tee -a "$log") 2>&1

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
export TRITON_CACHE_DIR="$root/runtime-tmp/triton-cache-node${node_rank}"
if (( leader )); then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

master_addr=$(python - <<'PY'
import socket
from mpi4py import MPI

comm = MPI.COMM_WORLD
address = socket.gethostbyname(socket.gethostname()) if comm.Get_rank() == 0 else None
print(comm.bcast(address, root=0))
PY
)

sync_status() {
    local local_code=$1
    LOCAL_CODE=$local_code python - <<'PY'
import os
from mpi4py import MPI

code = MPI.COMM_WORLD.allreduce(int(os.environ["LOCAL_CODE"]), op=MPI.MAX)
print(code)
PY
}

run_checked() {
    local phase=$1
    shift
    local code=0
    if (( leader )); then
        echo "=== START ${phase} node=${node_rank} host=$(hostname) ==="
        set +e
        "$@"
        code=$?
        set -e
    fi
    local global_code
    global_code=$(sync_status "$code")
    echo "=== END ${phase} mpi_rank=${world_rank} local_code=${code} global_code=${global_code} ==="
    if (( global_code != 0 )); then
        return "$global_code"
    fi
}

if (( leader )); then
    {
        echo "world_rank=$world_rank"
        echo "local_rank=$local_rank"
        echo "node_rank=$node_rank"
        echo "nnodes=$nnodes"
        echo "hostname=$(hostname)"
        echo "master_addr=$master_addr"
        echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
        nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
        nvidia-smi topo -m
    } >"$run_root/topology/node${node_rank}.txt"
fi

run_checked build_datasets \
    make -C "$root/third_party/Megatron-LM/megatron/core/datasets"

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
    --transformer-impl local
    --timeout-seconds 14400
)

run_checked preflight env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
    python -m torch.distributed.run \
        --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank" \
        --master-addr="$master_addr" --master-port=29616 \
        "$root/scripts/benchmark_multinode_collectives.py" \
        --output-dir "$run_root/preflight"

run_checked local_dp8 \
    python -m torch.distributed.run --standalone --nproc-per-node=8 \
        "$root/scripts/benchmark_muon_state_communication.py" \
        --output-dir "$run_root/local-dp8-node${node_rank}" \
        --models 4.9b \
        --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 8 \
        --global-batch-size 8 \
        "${common[@]}"

run_checked cross_dp8 env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.run \
        --nnodes=2 --nproc-per-node=4 --node-rank="$node_rank" \
        --master-addr="$master_addr" --master-port=29618 \
        "$root/scripts/benchmark_muon_state_communication.py" \
        --output-dir "$run_root/cross-dp8" \
        --models 4.9b \
        --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 8 \
        --global-batch-size 8 \
        "${common[@]}"

run_checked cross_dp16_narrow \
    python -m torch.distributed.run \
        --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank" \
        --master-addr="$master_addr" --master-port=29620 \
        "$root/scripts/benchmark_muon_state_communication.py" \
        --output-dir "$run_root/cross-dp16-narrow" \
        --models 4.9b \
        --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 16 \
        --global-batch-size 16 \
        "${common[@]}"

run_checked cross_dp16_wide \
    python -m torch.distributed.run \
        --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank" \
        --master-addr="$master_addr" --master-port=29622 \
        "$root/scripts/benchmark_muon_state_communication.py" \
        --output-dir "$run_root/cross-dp16-wide" \
        --models 5.0b-wide \
        --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 16 \
        --global-batch-size 16 \
        "${common[@]}"

run_checked cross_pp2_dp8 \
    python -m torch.distributed.run \
        --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank" \
        --master-addr="$master_addr" --master-port=29624 \
        "$root/scripts/benchmark_muon_state_communication.py" \
        --output-dir "$run_root/cross-pp2-dp8" \
        --models 9.9b-pp2 \
        --tensor-parallel-size 1 --pipeline-parallel-size 2 --data-parallel-size 8 \
        --global-batch-size 8 \
        --use-tp-pp-dp-mapping \
        "${common[@]}"

if [[ "$world_rank" == 0 ]]; then
    echo "=== MULTINODE 16-GPU RESULTS ==="
    find "$run_root" -name results.md -print -exec cat {} \;
    echo "RUN_ROOT=$run_root"
fi
