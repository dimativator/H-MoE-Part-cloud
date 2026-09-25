#!/usr/bin/env bash
set -euo pipefail

# Match the 75,457 * 318.8 ms Muon FP8-state step budget. The measured
# 500M BS32 step times set each optimizer's horizon; decay is 7,546 steps.
OPTIMIZER=${OPTIMIZER:?Set OPTIMIZER=slim_adam or frugal}
case "${OPTIMIZER}" in
    slim_adam)
        ITERATIONS=90333
        OPT_NAME=slim_adam
        LR=5e-4
        WEIGHT_DECAY=0.1
        RESULTS_DIR=${RESULTS_DIR:-/home/jovyan/dimativator/500m-time-matched-20260925/slim_adam}
        EXTRA_OPT_ARGS=()
        ;;
    frugal)
        ITERATIONS=91258
        OPT_NAME=coord_adamw
        LR=1e-3
        WEIGHT_DECAY=1e-4
        RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/frugal}
        EXTRA_OPT_ARGS=(--density 0.25 --update_gap 50 --coord_choice columns)
        ;;
    *) echo "Unsupported OPTIMIZER=${OPTIMIZER}" >&2; exit 2 ;;
esac

NPROC_PER_NODE=4
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
WANDB_GROUP=500M_time_matched_muon_fp8_1xC_20260925
EXPERIMENT_NAME=llama500M_${OPTIMIZER}_bf16_time_matched_4gpu
EXPERIMENT_DIR=${RESULTS_DIR}/${WANDB_GROUP}/${EXPERIMENT_NAME}
METRICS_JSONL=${EXPERIMENT_DIR}/metrics.jsonl
RESUME_FROM=${RESUME_FROM:-}

MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
MPI_RANK=${OMPI_COMM_WORLD_RANK:-0}
MPI_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
if (( MPI_SIZE > 1 && MPI_SIZE != NPROC_PER_NODE )); then
    echo "Expected ${NPROC_PER_NODE} MPI ranks, got ${MPI_SIZE}" >&2
    exit 2
fi
if [[ ! -f "${DATASETS_DIR}/packed_metadata.json" ]]; then
    echo "Packed FineWeb missing from ${DATASETS_DIR}" >&2
    exit 3
fi
DATASETS_DIR="${DATASETS_DIR}" ITERATIONS="${ITERATIONS}" python - <<'PY'
import json
import os
from pathlib import Path

metadata = json.loads((Path(os.environ['DATASETS_DIR']) / 'packed_metadata.json').read_text())
assert metadata['format'] == 'packed_fineweb_h200_v3', metadata['format']
assert int(metadata['iterations']) >= int(os.environ['ITERATIONS'])
assert int(metadata['world_size']) == 2
assert int(metadata['batch_size']) == 16
print('DATA_COVERAGE_CHECK=ok', metadata['iterations'], flush=True)
PY
if [[ -n "${RESUME_FROM}" ]]; then
    test -f "${RESUME_FROM}/main.pt"
    for rank in 0 1 2 3; do test -f "${RESUME_FROM}/worker_${rank}.pt"; done
fi
mkdir -p "${RESULTS_DIR}"
available_bytes=$(df -B1 --output=avail "${RESULTS_DIR}" | tail -n 1 | tr -d ' ')
if (( available_bytes < 8000000000 )); then
    echo "Insufficient space for a rotating checkpoint: ${available_bytes} bytes" >&2
    exit 4
fi

mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/wandb" "${EVAL_CACHE_DIR}"
exec > >(tee -a "${RESULTS_DIR}/logs/${EXPERIMENT_NAME}_rank${MPI_RANK}.log") 2>&1
echo "RUN_START=$(date --iso-8601=seconds) OPTIMIZER=${OPTIMIZER} RANK=${MPI_RANK}/${MPI_SIZE}"
echo "ITERATIONS=${ITERATIONS} DECAY_STEPS=7546 WARMUP_STEPS=2000 CHECKPOINT_INTERVAL=5000"
echo "DATASETS_DIR=${DATASETS_DIR} RESULTS_DIR=${RESULTS_DIR} RESUME_FROM=${RESUME_FROM:-none}"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

PYTHON_BIN=$(command -v python)
"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.__version__.startswith('2.9.1'), torch.__version__
assert torch.cuda.is_available()
print('ENVIRONMENT_CHECK=ok', torch.__version__, flush=True)
PY

WSD_FRACT_DECAY=$("${PYTHON_BIN}" - "${ITERATIONS}" <<'PY'
import sys
iterations = int(sys.argv[1])
fraction = (7546.5 / iterations)
assert int(iterations * fraction) == 7546
print(repr(fraction))
PY
)

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

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export FINEWEB_LOG_DATA_HASHES=1
export WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey
export WANDB_MODE=offline WANDB_DIR="${RESULTS_DIR}/wandb"
export TRITON_CACHE_DIR="/tmp/triton-500m-time-matched-${OPTIMIZER}-rank${MPI_RANK}-$$"
mkdir -p "${TRITON_CACHE_DIR}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then RESUME_ARGS=(--resume-from "${RESUME_FROM}"); fi
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
    --opt "${OPT_NAME}" --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" \
    --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    "${EXTRA_OPT_ARGS[@]}" \
    --scheduler wsd --warmup-steps 2000 --iterations "${ITERATIONS}" \
    --wsd-fract-decay "${WSD_FRACT_DECAY}" --wsd-final-lr-scale 0 --decay-type cosine \
    --batch-size 8 --acc-steps 16 --eval-batch-size 32 \
    --eval-interval 500 --eval-batches 32 --log-interval 50 \
    --downstream-eval-enabled --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --latest-ckpt-interval 5000 \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb --wandb-project fp8-pretrain --wandb-group "${WANDB_GROUP}" \
    --wandb-tags fineweb bf16 native_states 500M 1xC time_matched "${OPTIMIZER}" \
    --metrics-jsonl "${METRICS_JSONL}" \
    "${RESUME_ARGS[@]}"

if (( MPI_RANK == 0 )); then
    echo "TIME_MATCHED_COMPLETE optimizer=${OPTIMIZER} iter=${ITERATIONS}"
fi
