#!/usr/bin/env bash
set -euo pipefail

PRECISION=${PRECISION:?Set PRECISION to bf16, fp8_act, fp8_full, or w16a16g32_fp8_states}
case "${PRECISION}" in
    bf16|fp8_act|fp8_full|w16a16g32_fp8_states) ;;
    *) echo "Unsupported PRECISION=${PRECISION}" >&2; exit 2 ;;
esac
MODE=${MODE:-full}
case "${MODE}" in smoke|full) ;; *) echo "MODE must be smoke or full" >&2; exit 2 ;; esac

NPROC_PER_NODE=4
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
RESULTS_DIR=${RESULTS_DIR:-/home/jovyan/dimativator/500m-slimadam-precisions-20261002}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
TRUNK_GROUP=2xChinchilla_500M_slimadam_wd1e-4_precisions_4gpu
DECAY_GROUP=1xChinchilla_decay_500M_slimadam_wd1e-4_precisions_4gpu
TRUNK_NAME=llama500M_slim_adam_${PRECISION}_wd1e-4_2xC_warmup2000_4gpu
DECAY_NAME=llama500M_slim_adam_${PRECISION}_wd1e-4_1xC_decay_warmup2000_4gpu
TRUNK_DIR=${RESULTS_DIR}/${TRUNK_GROUP}/${TRUNK_NAME}
DECAY_DIR=${RESULTS_DIR}/${DECAY_GROUP}/${DECAY_NAME}
SOURCE_CKPT=${TRUNK_DIR}/ckpts/67911
REMOTE_NAME=${TRUNK_NAME}-iter-67911
MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
MPI_RANK=${OMPI_COMM_WORLD_RANK:-0}
MPI_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
if (( MPI_SIZE > 1 && MPI_SIZE != NPROC_PER_NODE )); then
    echo "Expected ${NPROC_PER_NODE} MPI ranks, got ${MPI_SIZE}" >&2; exit 2
fi

mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/wandb" "${EVAL_CACHE_DIR}"
exec > >(tee -a "${RESULTS_DIR}/logs/${TRUNK_NAME}_${MODE}_rank${MPI_RANK}.log") 2>&1
echo "RUN_START=$(date --iso-8601=seconds) MODE=${MODE} PRECISION=${PRECISION} WARMUP_STEPS=2000 RANK=${MPI_RANK}/${MPI_SIZE}"
df -h "${RESULTS_DIR}" /home/jovyan /workspace-SR006.nfs3

PYTHON_BIN=$(command -v python)
DATASETS_DIR="${DATASETS_DIR}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
import torch
metadata = json.loads((Path(os.environ['DATASETS_DIR']) / 'packed_metadata.json').read_text())
assert metadata['format'] == 'packed_fineweb_h200_v3'
assert int(metadata['iterations']) >= 150914
assert int(metadata['world_size']) == 2
assert int(metadata['batch_size']) == 16
assert torch.__version__.startswith('2.9.1')
assert torch.cuda.is_available()
print('ENVIRONMENT_AND_DATA_CHECK=ok', flush=True)
PY

if [[ "${MODE}" == full ]]; then
    [[ -n "${BRAIN_LAB_RELAY_KEY_B64:-}${BRAIN_LAB_RELAY_KEY_B64_0:-}" ]] || { echo 'Missing relay credential' >&2; exit 3; }
    available_bytes=$(df -B1 --output=avail "${RESULTS_DIR}" | tail -n 1 | tr -d ' ')
    (( available_bytes >= 6500000000 )) || { echo "Insufficient checkpoint space: ${available_bytes}" >&2; exit 4; }
fi

if (( MPI_SIZE > 1 )); then
    export RANK=${RANK:-${MPI_RANK}} WORLD_SIZE=${WORLD_SIZE:-${MPI_SIZE}} LOCAL_RANK=${LOCAL_RANK:-${MPI_LOCAL_RANK}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)} MASTER_PORT=${MASTER_PORT:-29500}
    TRAIN_LAUNCHER=("${PYTHON_BIN}")
else
    TRAIN_LAUNCHER=("$(command -v torchrun)" --standalone --nproc_per_node=4)
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export FINEWEB_LOG_DATA_HASHES=1
export WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey WANDB_MODE=offline WANDB_DIR="${RESULTS_DIR}/wandb"
export TRITON_CACHE_DIR="/tmp/triton-500m-slimadam-${PRECISION}-${MODE}-rank${MPI_RANK}-$$"
mkdir -p "${TRITON_CACHE_DIR}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PRECISION_ARGS=()
case "${PRECISION}" in
    fp8_act|fp8_full)
        PRECISION_ARGS+=(--fp8 --fp8-fabit E4M3 --fp8-fwbit E4M3 --fp8-babit E5M2 --fp8-bwbit E5M2 --fp8-group-size 16)
        ;;
esac
case "${PRECISION}" in
    fp8_full|w16a16g32_fp8_states)
        PRECISION_ARGS+=(--fp8-optim --fp8-qgroup-size 128 --fp8-first-order-bit E4M3 --fp8-second-order-bit E4M3 --fp8-expansion expand)
        ;;
esac

COMMON_ARGS=(
    --distributed-backend nccl --seed 0 --data-seed 1337
    --dataset fineweb --datasets-dir "${DATASETS_DIR}"
    --fineweb-replay-world-size 1 --fineweb-replay-layout concat
    --eval-cache-dir "${EVAL_CACHE_DIR}" --sequence-length 1024 --streaming --workers 8
    --model llama --n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256
    --dtype bfloat16 --opt slim_adam --lr 5e-4 --weight-decay 1e-4
    --beta1 0.9 --beta2 0.99 --grad-clip 1.0
    --batch-size 8 --acc-steps 16 --eval-batch-size 32
    --results-base-folder "${RESULTS_DIR}" "${PRECISION_ARGS[@]}"
)

if [[ "${MODE}" == smoke ]]; then
    "${TRAIN_LAUNCHER[@]}" src/main.py "${COMMON_ARGS[@]}" \
        --experiment-name "smoke_${TRUNK_NAME}" \
        --scheduler none --warmup-steps 0 --iterations 2 \
        --eval-interval 2 --eval-batches 1 --log-interval 1 --no-local-save
    if (( MPI_RANK == 0 )); then echo "SLIMADAM_PRECISION_SMOKE_COMPLETE precision=${PRECISION}"; fi
    exit 0
fi

if [[ -e "${TRUNK_DIR}/metrics.jsonl" || -e "${DECAY_DIR}/metrics.jsonl" ]]; then
    echo "Refusing to overwrite an existing run: ${TRUNK_DIR}" >&2; exit 5
fi

"${TRAIN_LAUNCHER[@]}" src/main.py "${COMMON_ARGS[@]}" \
    --experiment-name "${TRUNK_NAME}" \
    --scheduler wsd --warmup-steps 2000 --iterations 150914 \
    --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine \
    --eval-interval 500 --eval-batches 32 --log-interval 50 \
    --downstream-eval-enabled --downstream-eval-interval 2000 --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --inter-ckpts 67911 --latest-ckpt-interval 0 \
    --upload-inter-ckpts-to brain_lab --delete-local-inter-ckpts-after-upload \
    --wandb --wandb-project fp8-pretrain --wandb-group "${TRUNK_GROUP}" \
    --wandb-tags fineweb 2xChinchilla 500M slim_adam wd1e-4 warmup2000 "${PRECISION}" \
    --metrics-jsonl "${TRUNK_DIR}/metrics.jsonl"

if (( MPI_RANK == 0 )); then
    echo "SLIMADAM_2XC_COMPLETE precision=${PRECISION} iter=150914"
    if [[ -e "${SOURCE_CKPT}" ]]; then
        echo 'Local pre-decay checkpoint remained after relay; refusing to continue' >&2; exit 6
    fi
    "${PYTHON_BIN}" scripts/cloud/relay_brain_lab_checkpoint.py download "${SOURCE_CKPT}" "${REMOTE_NAME}"
    touch "${SOURCE_CKPT}/.relay_ready"
else
    for (( attempt=0; attempt<720; attempt++ )); do
        [[ -f "${SOURCE_CKPT}/.relay_ready" ]] && break
        sleep 10
    done
    [[ -f "${SOURCE_CKPT}/.relay_ready" ]] || { echo 'Timed out waiting for verified relay download' >&2; exit 7; }
fi

"${TRAIN_LAUNCHER[@]}" src/main.py "${COMMON_ARGS[@]}" \
    --experiment-name "${DECAY_NAME}" --resume-from "${SOURCE_CKPT}" --decay-from-checkpoint \
    --scheduler wsd --warmup-steps 0 --iterations 75457 \
    --wsd-fract-decay 1.0 --wsd-final-lr-scale 0 --decay-type cosine \
    --eval-interval 500 --eval-batches 32 --log-interval 50 \
    --downstream-eval-enabled --downstream-eval-interval 2000 --downstream-task-group basic_v2 \
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103 \
    --latest-ckpt-interval 0 --no-local-save \
    --wandb --wandb-project fp8-pretrain --wandb-group "${DECAY_GROUP}" \
    --wandb-tags fineweb 1xChinchilla decay 500M slim_adam wd1e-4 warmup2000 "${PRECISION}" \
    --metrics-jsonl "${DECAY_DIR}/metrics.jsonl"

if (( MPI_RANK == 0 )); then
    echo "SLIMADAM_1XC_DECAY_COMPLETE precision=${PRECISION} iter=75457"
    rm -f "${SOURCE_CKPT}/.relay_ready"
    for required in main.pt worker_0.pt worker_1.pt worker_2.pt worker_3.pt; do
        [[ -s "${SOURCE_CKPT}/${required}" ]] || { echo "Missing local checkpoint file ${required}" >&2; exit 8; }
        rm -- "${SOURCE_CKPT}/${required}"
    done
    rmdir -- "${SOURCE_CKPT}"
    echo "LOCAL_PRE_DECAY_CHECKPOINT_REMOVED=${SOURCE_CKPT}"
fi
