#!/usr/bin/env bash
set -euo pipefail

# Two queues, six independent 1xC trainings each. Never resume or write checkpoints.
QUEUE_ID=${QUEUE_ID:?Set QUEUE_ID=A or B}
MODE=${MODE:-full}
case "${QUEUE_ID}" in A) OPTIMIZERS=(frugal galore) ;; B) OPTIMIZERS=(frugal_mm apollo) ;; *) exit 2 ;; esac
SKIP_OPTIMIZER=${SKIP_OPTIMIZER:-}
case "${SKIP_OPTIMIZER}" in ''|frugal|galore|frugal_mm|apollo) ;; *) echo 'Invalid SKIP_OPTIMIZER' >&2; exit 2 ;; esac
case "${MODE}" in smoke|full|probe) ;; *) exit 2 ;; esac
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
RESULTS_DIR=${RESULTS_DIR:-/home/jovyan/dimativator/500m-bf16-wd-tuning-20261006}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
GROUP=500M_BF16_WD_tuning_1xC_20261006
MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
MPI_RANK=${OMPI_COMM_WORLD_RANK:-0}
MPI_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey WANDB_MODE=online
export WANDB_DIR="${RESULTS_DIR}/wandb" WANDB_INIT_TIMEOUT=120
[[ -n "${WANDB_API_KEY:-}" ]] || { echo 'Missing W&B credential' >&2; exit 3; }
if [[ "${MODE}" == probe ]]; then
    python - <<'PY'
import os
import wandb
api = wandb.Api(timeout=30)
run = api.run('andrey/fp8-pretrain/frugalmm-6ad93294010c')
assert run.config['weight_decay'] == 0.1
print('WANDB_AUTH_OK host=https://wandb-radfan.ru entity=andrey project=fp8-pretrain')
PY
    df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
    exit 0
fi
(( MPI_SIZE == 1 || MPI_SIZE == 2 )) || { echo 'Exactly two GPU ranks required' >&2; exit 2; }
mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/markers" "${WANDB_DIR}" "${EVAL_CACHE_DIR}"
exec > >(tee -a "${RESULTS_DIR}/logs/queue_${QUEUE_ID}_${MODE}_rank${MPI_RANK}.log") 2>&1
echo "WD_QUEUE_START=$(date -u --iso-8601=seconds) QUEUE=${QUEUE_ID} MODE=${MODE} WARMUP_STEPS=2000"
df -h "${RESULTS_DIR}"
DATASETS_DIR="${DATASETS_DIR}" python - <<'PY'
import json
import os
from pathlib import Path
import torch
m = json.loads((Path(os.environ['DATASETS_DIR']) / 'packed_metadata.json').read_text())
assert m['format'] == 'packed_fineweb_h200_v3'
assert int(m['iterations']) >= 75457
assert int(m['world_size']) == 2 and int(m['batch_size']) == 16
assert torch.__version__.startswith('2.9.1') and torch.cuda.is_available()
print('ENVIRONMENT_AND_DATA_CHECK=ok')
PY
if (( MPI_SIZE > 1 )); then
    export RANK=${RANK:-${MPI_RANK}} WORLD_SIZE=${WORLD_SIZE:-${MPI_SIZE}} LOCAL_RANK=${LOCAL_RANK:-${MPI_LOCAL_RANK}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)} MASTER_PORT=${MASTER_PORT:-29500}
    TRAIN_LAUNCHER=("$(command -v python)")
else
    TRAIN_LAUNCHER=("$(command -v torchrun)" --standalone --nproc_per_node=2)
fi
export FINEWEB_LOG_DATA_HASHES=1
export TRITON_CACHE_DIR="/tmp/triton-wd500-${QUEUE_ID}-${MODE}-rank${MPI_RANK}-$$"
mkdir -p "${TRITON_CACHE_DIR}"
CURRENT_FAILURE_MARKER=
on_exit() {
    local code=$?
    if (( code != 0 )) && [[ -n "${CURRENT_FAILURE_MARKER}" ]]; then
        printf '%s\n' "${code}" > "${CURRENT_FAILURE_MARKER}"
        echo "WD_QUEUE_FAILED queue=${QUEUE_ID} code=${code} utc=$(date -u --iso-8601=seconds)"
    fi
}
trap on_exit EXIT
for OPTIMIZER in "${OPTIMIZERS[@]}"; do
    if [[ "${OPTIMIZER}" == "${SKIP_OPTIMIZER}" ]]; then
        echo "SKIP_USER_OPTIMIZER=${OPTIMIZER}"
        continue
    fi
    OPT_ARGS=()
    case "${OPTIMIZER}" in
        frugal) OPT=coord_adamw; LR=1e-3; BETA2=0.999 ;;
        galore) OPT=galore_adamw; LR=1e-3; BETA2=0.999; OPT_ARGS+=(--proj_type svd --proj_side std) ;;
        frugal_mm) OPT=coord_muon; LR=2e-3; BETA2=0.99; OPT_ARGS+=(--non_proj_opt adamw --momentum 0.95 --nesterov --muon_ns_steps 5) ;;
        apollo) OPT=apollo_adamw; LR=2e-3; BETA2=0.999; OPT_ARGS+=(--apollo_proj random --apollo_scale_type channel --apollo_scale 1 --proj_side std) ;;
    esac
    WDS=(1e-2 1e-3 1e-4)
    if [[ "${MODE}" == smoke ]]; then WDS=(1e-2); fi
    for WEIGHT_DECAY in "${WDS[@]}"; do
        NAME="llama500M_${OPTIMIZER}_bf16_wd${WEIGHT_DECAY}_1xC_warmup2000_2gpu_20261006"
        RUN_ID="wd500-${OPTIMIZER}-${WEIGHT_DECAY}-20261006"
        TARGET=75457
        SCHEDULER=wsd
        WARMUP=2000
        EVAL_INTERVAL=500
        EVAL_BATCHES=32
        LOG_INTERVAL=50
        EVAL_ARGS=(--downstream-eval-enabled --downstream-eval-interval 2000 --downstream-task-group basic_v2
                   --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103)
        if [[ "${MODE}" == smoke ]]; then
            NAME="smoke_${NAME}"; RUN_ID="smoke-${RUN_ID}"
            TARGET=2; SCHEDULER=none; WARMUP=0; EVAL_INTERVAL=2; EVAL_BATCHES=1; LOG_INTERVAL=1; EVAL_ARGS=()
        fi
        METRICS=${RESULTS_DIR}/${GROUP}/${NAME}/metrics.jsonl
        DONE=${RESULTS_DIR}/markers/${NAME}.done
        CURRENT_FAILURE_MARKER=${RESULTS_DIR}/markers/${NAME}.failed.rank${MPI_RANK}
        if [[ -f "${DONE}" ]]; then
            echo "SKIP_VERIFIED_COMPLETION=${NAME}"
            continue
        fi
        # The marker written by rank 0 proves both local and W&B exact finals.
        [[ ! -e "${METRICS}" ]] || { echo "Refusing to overwrite existing metrics: ${METRICS}" >&2; exit 5; }
        available=$(df -B1 --output=avail "${RESULTS_DIR}" | tail -n 1 | tr -d ' ')
        (( available >= 1000000000 )) || { echo 'Insufficient log headroom' >&2; exit 4; }
        export WANDB_RUN_ID="${RUN_ID}" WANDB_RESUME=never
        echo "WD_RUN_START optimizer=${OPTIMIZER} wd=${WEIGHT_DECAY} lr=${LR} beta2=${BETA2} target=${TARGET} wandb_id=${RUN_ID} utc=$(date -u --iso-8601=seconds)"
        "${TRAIN_LAUNCHER[@]}" src/main.py \
            --distributed-backend nccl --seed 0 --data-seed 1337 \
            --dataset fineweb --datasets-dir "${DATASETS_DIR}" --fineweb-replay-world-size 1 --fineweb-replay-layout concat \
            --eval-cache-dir "${EVAL_CACHE_DIR}" --sequence-length 1024 --streaming --workers 8 \
            --model llama --n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256 \
            --dtype bfloat16 --opt "${OPT}" --lr "${LR}" --weight-decay "${WEIGHT_DECAY}" \
            --beta1 0.9 --beta2 "${BETA2}" --grad-clip 1.0 \
            --density 0.25 --update_gap 50 --coord_choice columns --inactive_update_rule sign_sgd \
            --batch-size 16 --acc-steps 8 --eval-batch-size 32 \
            --scheduler "${SCHEDULER}" --warmup-steps "${WARMUP}" --iterations "${TARGET}" \
            --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine \
            --eval-interval "${EVAL_INTERVAL}" --eval-batches "${EVAL_BATCHES}" --log-interval "${LOG_INTERVAL}" \
            --latest-ckpt-interval 0 --permanent-ckpt-interval 0 --no-local-save \
            --experiment-name "${NAME}" --results-base-folder "${RESULTS_DIR}" --metrics-jsonl "${METRICS}" \
            --wandb --wandb-project fp8-pretrain --wandb-group "${GROUP}" \
            --wandb-tags 500M BF16 WD_tuning 1xChinchilla "${OPTIMIZER}" "wd${WEIGHT_DECAY}" warmup2000 \
            "${OPT_ARGS[@]}" "${EVAL_ARGS[@]}"
        if (( MPI_RANK == 0 )); then
            python scripts/cloud/verify_500m_bf16_wd_tuning.py \
                --metrics "${METRICS}" --run-id "${RUN_ID}" --target "${TARGET}" --opt "${OPT}" --wd "${WEIGHT_DECAY}" --lr "${LR}"
            touch "${DONE}"
            echo "WD_TUNING_COMPLETE optimizer=${OPTIMIZER} wd=${WEIGHT_DECAY} iter=${TARGET} wandb_id=${RUN_ID}"
        else
            for (( attempt=0; attempt<300; attempt++ )); do
                [[ ! -f "${DONE}" ]] || break
                [[ ! -f "${RESULTS_DIR}/markers/${NAME}.failed.rank0" ]] || { echo 'Rank-0 verification failed' >&2; exit 6; }
                sleep 2
            done
            [[ -f "${DONE}" ]] || { echo 'Timed out waiting for verified final' >&2; exit 6; }
        fi
    done
done
CURRENT_FAILURE_MARKER=
echo "WD_QUEUE_COMPLETE queue=${QUEUE_ID} mode=${MODE} utc=$(date -u --iso-8601=seconds)"
