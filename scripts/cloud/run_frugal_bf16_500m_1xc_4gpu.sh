#!/usr/bin/env bash
set -euo pipefail

# Standard Frugal AdamW, BF16 compute and states, 500M Llama, 1xC.
# The DDP backend converts pre-DDP batch 8 x accumulation 16 into
# per-rank batch 8 x accumulation 4 on four GPUs (global batch 128).

NPROC_PER_NODE=4
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/frugal-bf16-500m-1xc-wd1e-4-4gpu-20260920}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
EXPERIMENT_NAME=llama500M_frugal_adamw_bf16_wd1e-4_1xC_4gpu
WANDB_GROUP=1xChinchilla_500M_frugal_bf16_wd1e-4_4gpu_cloud
METRICS_JSONL=${RESULTS_DIR}/${WANDB_GROUP}/${EXPERIMENT_NAME}/metrics.jsonl

MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
MPI_RANK=${OMPI_COMM_WORLD_RANK:-0}
MPI_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
if (( MPI_SIZE > 1 && MPI_SIZE != NPROC_PER_NODE )); then
    echo "Expected ${NPROC_PER_NODE} MPI ranks, got ${MPI_SIZE}" >&2
    exit 2
fi
if [[ ! -f "${DATASETS_DIR}/packed_metadata.json" ]]; then
    echo "Packed FineWeb is missing: ${DATASETS_DIR}" >&2
    exit 3
fi

mkdir -p "${RESULTS_DIR}/logs" "${EVAL_CACHE_DIR}"
exec > >(tee -a "${RESULTS_DIR}/logs/${EXPERIMENT_NAME}_rank${MPI_RANK}.log") 2>&1
echo "EXPERIMENT=${EXPERIMENT_NAME} MPI_RANK=${MPI_RANK}/${MPI_SIZE} DATE=$(date --iso-8601=seconds)"
echo "DATASETS_DIR=${DATASETS_DIR} METRICS_JSONL=${METRICS_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv || true

PYTHON_BIN=$(command -v python)
if (( MPI_SIZE > 1 )); then
    export RANK=${RANK:-${MPI_RANK}}
    export WORLD_SIZE=${WORLD_SIZE:-${MPI_SIZE}}
    export LOCAL_RANK=${LOCAL_RANK:-${MPI_LOCAL_RANK}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
    export MASTER_PORT=${MASTER_PORT:-29500}
    TRAIN_LAUNCHER=("${PYTHON_BIN}")
else
    TRAIN_LAUNCHER=("$(command -v torchrun)" --standalone --nproc_per_node=4)
fi

"${PYTHON_BIN}" - <<'PY'
import torch

assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.cuda.is_available()
print("ENVIRONMENT_CHECK=ok", torch.__version__, torch.version.cuda)
PY

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FINEWEB_LOG_DATA_HASHES=1
export TRITON_CACHE_DIR="/tmp/triton-frugal-bf16-1xc-rank${MPI_RANK}-$$"
mkdir -p "${TRITON_CACHE_DIR}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"${TRAIN_LAUNCHER[@]}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "${EXPERIMENT_NAME}" \
    --seed 0 --data-seed 1337 \
    --dataset fineweb --datasets-dir "${DATASETS_DIR}" \
    --fineweb-replay-world-size 1 --fineweb-replay-layout concat \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length 1024 --streaming --workers 8 \
    --model llama --n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256 \
    --dtype bfloat16 \
    --opt coord_adamw --lr 1e-3 --weight-decay 1e-4 \
    --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    --density 0.25 --update_gap 50 --coord_choice columns \
    --scheduler wsd --warmup-steps 7000 --iterations 75457 \
    --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine \
    --batch-size 8 --acc-steps 16 --eval-batch-size 32 \
    --eval-interval 500 --eval-batches 32 --log-interval 50 \
    --downstream-eval-enabled --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --latest-ckpt-interval 0 --no-local-save \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb-group "${WANDB_GROUP}" --metrics-jsonl "${METRICS_JSONL}"
