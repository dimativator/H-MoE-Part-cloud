#!/usr/bin/env bash
set -euo pipefail

# Three independent WD jobs with 2000 warmup steps: 2xC trunk, then 1xC decay from step 67911.
WEIGHT_DECAY=${WEIGHT_DECAY:?Set WEIGHT_DECAY to 1e-2, 1e-3, or 1e-4}
case "${WEIGHT_DECAY}" in
    1e-2|1e-3|1e-4) ;;
    *) echo "Unsupported WEIGHT_DECAY=${WEIGHT_DECAY}" >&2; exit 2 ;;
esac
MODE=${MODE:-full}
case "${MODE}" in
    smoke|full) ;;
    *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac

NPROC_PER_NODE=2
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
RESULTS_DIR=${RESULTS_DIR:?Set RESULTS_DIR to a dedicated experiment directory}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
TRUNK_GROUP=2xChinchilla_500M_slimadam_bf16_wd_sweep_2gpu_cloud
DECAY_GROUP=1xChinchilla_decay_500M_slimadam_bf16_wd_sweep_2gpu_cloud
TRUNK_NAME=llama500M_slim_adam_bf16_wd${WEIGHT_DECAY}_2xC_warmup2000_2gpu
DECAY_NAME=llama500M_slim_adam_bf16_wd${WEIGHT_DECAY}_1xC_decay_warmup2000_2gpu
TRUNK_DIR=${RESULTS_DIR}/${TRUNK_GROUP}/${TRUNK_NAME}
DECAY_DIR=${RESULTS_DIR}/${DECAY_GROUP}/${DECAY_NAME}
SOURCE_CKPT=${TRUNK_DIR}/ckpts/67911

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

mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/wandb" "${EVAL_CACHE_DIR}"
exec > >(tee -a "${RESULTS_DIR}/logs/${TRUNK_NAME}_${MODE}_rank${MPI_RANK}.log") 2>&1
echo "RUN_START=$(date --iso-8601=seconds) MODE=${MODE} WD=${WEIGHT_DECAY} WARMUP_STEPS=2000 RANK=${MPI_RANK}/${MPI_SIZE}"
echo "RESULTS_DIR=${RESULTS_DIR} DATASETS_DIR=${DATASETS_DIR}"
df -h "${RESULTS_DIR}" /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

PYTHON_BIN=$(command -v python)
DATASETS_DIR="${DATASETS_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import torch

metadata = json.loads((Path(os.environ['DATASETS_DIR']) / 'packed_metadata.json').read_text())
assert metadata['format'] == 'packed_fineweb_h200_v3', metadata['format']
assert int(metadata['iterations']) >= 150914, metadata['iterations']
assert int(metadata['world_size']) == 2
assert int(metadata['batch_size']) == 16
assert torch.__version__.startswith('2.9.1'), torch.__version__
assert torch.cuda.is_available()
print('ENVIRONMENT_AND_DATA_CHECK=ok', torch.__version__, metadata['iterations'], flush=True)
PY

if [[ "${MODE}" == full ]]; then
    available_bytes=$(df -B1 --output=avail "${RESULTS_DIR}" | tail -n 1 | tr -d ' ')
    if (( available_bytes < 7000000000 )); then
        echo "Insufficient space for pre-decay checkpoint: ${available_bytes} bytes" >&2
        exit 4
    fi
fi

if (( MPI_SIZE > 1 )); then
    export RANK=${RANK:-${MPI_RANK}}
    export WORLD_SIZE=${WORLD_SIZE:-${MPI_SIZE}}
    export LOCAL_RANK=${LOCAL_RANK:-${MPI_LOCAL_RANK}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)}
    export MASTER_PORT=${MASTER_PORT:-29500}
    TRAIN_LAUNCHER=("${PYTHON_BIN}")
else
    TRAIN_LAUNCHER=("$(command -v torchrun)" --standalone --nproc_per_node=2)
fi

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FINEWEB_LOG_DATA_HASHES=1
export WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey
export WANDB_MODE=offline WANDB_DIR="${RESULTS_DIR}/wandb"
export TRITON_CACHE_DIR="/tmp/triton-500m-slimadam-wd-${WEIGHT_DECAY}-${MODE}-rank${MPI_RANK}-$$"
mkdir -p "${TRITON_CACHE_DIR}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

COMMON_ARGS=(
    --distributed-backend nccl
    --seed 0 --data-seed 1337
    --dataset fineweb --datasets-dir "${DATASETS_DIR}"
    --fineweb-replay-world-size 1 --fineweb-replay-layout concat
    --eval-cache-dir "${EVAL_CACHE_DIR}"
    --sequence-length 1024 --streaming --workers 8
    --model llama --n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256
    --dtype bfloat16
    --opt slim_adam --lr 5e-4 --weight-decay "${WEIGHT_DECAY}"
    --beta1 0.9 --beta2 0.99 --grad-clip 1.0
    --batch-size 16 --acc-steps 8 --eval-batch-size 32
    --results-base-folder "${RESULTS_DIR}"
)

if [[ "${MODE}" == smoke ]]; then
    "${TRAIN_LAUNCHER[@]}" src/main.py \
        "${COMMON_ARGS[@]}" \
        --experiment-name "smoke_${TRUNK_NAME}" \
        --scheduler none --warmup-steps 0 --iterations 2 \
        --eval-interval 2 --eval-batches 1 --log-interval 1 \
        --no-local-save
    if (( MPI_RANK == 0 )); then echo "SLIMADAM_WD_SMOKE_COMPLETE wd=${WEIGHT_DECAY}"; fi
    exit 0
fi

if [[ -e "${TRUNK_DIR}/metrics.jsonl" || -e "${DECAY_DIR}/metrics.jsonl" ]]; then
    echo "Refusing to overwrite an existing run; inspect ${TRUNK_DIR} and ${DECAY_DIR}" >&2
    exit 5
fi

"${TRAIN_LAUNCHER[@]}" src/main.py \
    "${COMMON_ARGS[@]}" \
    --experiment-name "${TRUNK_NAME}" \
    --scheduler wsd --warmup-steps 2000 --iterations 150914 \
    --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine \
    --eval-interval 500 --eval-batches 32 --log-interval 50 \
    --downstream-eval-enabled --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --inter-ckpts 67911 --latest-ckpt-interval 0 \
    --wandb --wandb-project fp8-pretrain --wandb-group "${TRUNK_GROUP}" \
    --wandb-tags fineweb bf16 native_states 2xChinchilla 500M slim_adam wd_sweep warmup2000 \
    --metrics-jsonl "${TRUNK_DIR}/metrics.jsonl"

for required in main.pt worker_0.pt worker_1.pt; do
    if [[ ! -s "${SOURCE_CKPT}/${required}" ]]; then
        echo "Missing pre-decay checkpoint: ${SOURCE_CKPT}/${required}" >&2
        exit 6
    fi
done
if (( MPI_RANK == 0 )); then
    echo "SLIMADAM_2XC_COMPLETE wd=${WEIGHT_DECAY} iter=150914"
    echo "PRE_DECAY_CHECKPOINT=${SOURCE_CKPT}"
fi

"${TRAIN_LAUNCHER[@]}" src/main.py \
    "${COMMON_ARGS[@]}" \
    --experiment-name "${DECAY_NAME}" \
    --resume-from "${SOURCE_CKPT}" --decay-from-checkpoint \
    --scheduler wsd --warmup-steps 0 --iterations 75457 \
    --wsd-fract-decay 1.0 --wsd-final-lr-scale 0 --decay-type cosine \
    --eval-interval 500 --eval-batches 32 --log-interval 50 \
    --downstream-eval-enabled --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --latest-ckpt-interval 0 --no-local-save \
    --wandb --wandb-project fp8-pretrain --wandb-group "${DECAY_GROUP}" \
    --wandb-tags fineweb bf16 native_states 1xChinchilla decay 500M slim_adam wd_sweep warmup2000 \
    --metrics-jsonl "${DECAY_DIR}/metrics.jsonl"

if (( MPI_RANK == 0 )); then
    echo "SLIMADAM_1XC_DECAY_COMPLETE wd=${WEIGHT_DECAY} iter=75457"
fi
