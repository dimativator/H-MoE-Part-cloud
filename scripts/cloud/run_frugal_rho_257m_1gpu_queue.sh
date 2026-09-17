#!/usr/bin/env bash
set -euo pipefail

# Two independent Cloud.ru queues for the Llama-257M 1xC Frugal rho sweep.
# Frugal is the coordinate-projection AdamW variant (`coord_adamw`).

QUEUE_ID=${QUEUE_ID:-A}
RHO_LIST=${RHO_LIST:-}
MODE=${MODE:-full}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs2/dimativator/frugal-rho-257m-1xc-wsd-20260915}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-5000}
SAVE_CHECKPOINTS=${SAVE_CHECKPOINTS:-1}
ARCHIVE_INCOMPLETE=${ARCHIVE_INCOMPLETE:-0}
RUN_SEED=${SEED:-0}

case "${RESULTS_DIR}" in
    /workspace-SR006.nfs2/dimativator/frugal-rho-257m-1xc-*|\
    /workspace-SR006.nfs3/dimativator/frugal-rho-257m-1xc-*) ;;
    *)
        echo "Refusing checkpoint cleanup outside the dedicated experiment root: ${RESULTS_DIR}" >&2
        exit 2
        ;;
esac
if [[ -n "${RHO_LIST}" ]]; then
    read -r -a RHOS <<< "${RHO_LIST}"
else
    case "${QUEUE_ID}" in
        A) RHOS=(1.0 0.8 0.6 0.4 0.2 0.1) ;;
        B) RHOS=(0.9 0.7 0.5 0.3 0.15) ;;
        *) echo "QUEUE_ID must be A or B" >&2; exit 2 ;;
    esac
fi
for rho in "${RHOS[@]}"; do
    if [[ ! "${rho}" =~ ^(0\.[0-9]*[1-9][0-9]*|1(\.0+)?)$ ]]; then
        echo "Each RHO_LIST value must be in (0, 1]: ${rho}" >&2
        exit 2
    fi
done
case "${MODE}" in
    smoke|full) ;;
    *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac
case "${SAVE_CHECKPOINTS}" in
    0|1) ;;
    *) echo "SAVE_CHECKPOINTS must be 0 or 1" >&2; exit 2 ;;
esac
case "${ARCHIVE_INCOMPLETE}" in
    0|1) ;;
    *) echo "ARCHIVE_INCOMPLETE must be 0 or 1" >&2; exit 2 ;;
esac

mkdir -p "${RESULTS_DIR}/logs" "${EVAL_CACHE_DIR}"
QUEUE_LOG="${RESULTS_DIR}/logs/queue_${QUEUE_ID}_${MODE}.log"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "QUEUE_ID=${QUEUE_ID} MODE=${MODE} RHOS=${RHOS[*]}"
echo "DATASETS_DIR=${DATASETS_DIR} RESULTS_DIR=${RESULTS_DIR}"
echo "HOST=$(hostname) DATE=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

PYTHON_BIN=$(command -v python)
TORCHRUN_BIN=$(command -v torchrun)

"${PYTHON_BIN}" - <<'PY'
import torch
import torchao
import triton

assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), torch.version.cuda
assert torch.cuda.is_available()
assert torchao.__version__.startswith("0.15.0"), torchao.__version__
assert triton.__version__ == "3.5.1", triton.__version__
print("ENVIRONMENT_CHECK=ok", torch.__version__, torch.version.cuda)
PY

if [[ ! -f "${DATASETS_DIR}/packed_metadata.json" ]]; then
    echo "Packed H200 data is missing: ${DATASETS_DIR}" >&2
    exit 5
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton-frugal-rho-${QUEUE_ID}-${MODE}-$$"
mkdir -p "${TRITON_CACHE_DIR}"

remove_checkpoint_tree() {
    local target=$1
    case "${target}" in
        "${RESULTS_DIR}/"*/ckpts) ;;
        *) echo "Refusing unsafe checkpoint removal: ${target}" >&2; exit 11 ;;
    esac
    rm -rf -- "${target}"
}

archive_incomplete_experiment() {
    local target=$1
    local stamp
    [[ "${MODE}" == "full" && "${ARCHIVE_INCOMPLETE}" == "1" ]] || return
    [[ -f "${target}/metrics.jsonl" ]] || return
    case "${target}" in
        "${RESULTS_DIR}/"*) ;;
        *) echo "Refusing unsafe incomplete-run archive: ${target}" >&2; exit 12 ;;
    esac
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    mv -- "${target}" "${target}.interrupted.${stamp}"
    echo "ARCHIVED_INCOMPLETE=${target}.interrupted.${stamp}"
}

run_rho() {
    local rho=$1
    local suffix=${rho/./p}
    local experiment="llama257M_frugal_rho${suffix}_bf16_wsd_lr1e-3_wd0p1_1xC_1gpu"
    local group="1xChinchilla_257M_frugal_rho_wsd_cloud"
    local exp_dir="${RESULTS_DIR}/${group}/${experiment}"
    local marker="${RESULTS_DIR}/.${experiment}.${MODE}.done"
    [[ -f "${marker}" ]] && return

    local iterations=39250
    local warmup=2000
    local eval_interval=500
    local eval_batches=32
    local log_interval=50
    local save_args=(--latest-ckpt-interval "${LATEST_CKPT_INTERVAL}")
    local eval_args=(
        --downstream-eval-enabled --downstream-eval-interval 2000
        --downstream-task-group basic_v2
        --lm-eval-enabled --lm-eval-interval 2000
        --lm-eval-datasets wikitext103
    )
    if [[ "${MODE}" == "smoke" ]]; then
        experiment="smoke_${experiment}"
        exp_dir="${RESULTS_DIR}/smoke/${experiment}"
        iterations=3
        warmup=0
        eval_interval=3
        eval_batches=1
        log_interval=1
        save_args=(--no-local-save)
        eval_args=()
    elif [[ "${SAVE_CHECKPOINTS}" == "0" ]]; then
        save_args=(--no-local-save)
    fi

    archive_incomplete_experiment "${exp_dir}"

    echo "RUN_START queue=${QUEUE_ID} rho=${rho} experiment=${experiment}"
    "${TORCHRUN_BIN}" --standalone --nproc_per_node=1 src/main.py \
        --distributed-backend nccl \
        --experiment-name "${experiment}" \
        --seed "${RUN_SEED}" --data-seed 1337 \
        --dataset fineweb --datasets-dir "${DATASETS_DIR}" \
        --fineweb-replay-world-size 2 --fineweb-replay-layout concat \
        --eval-cache-dir "${EVAL_CACHE_DIR}" \
        --sequence-length 1024 --streaming --workers 8 \
        --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256 \
        --dtype bfloat16 \
        --opt coord_adamw --non_proj_opt adamw \
        --lr 1e-3 --weight-decay 0.1 --beta1 0.9 --beta2 0.999 \
        --grad-clip 1.0 --density "${rho}" --update_gap 50 \
        --coord_choice columns \
        --scheduler wsd --warmup-steps "${warmup}" --iterations "${iterations}" \
        --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine \
        --batch-size 32 --acc-steps 4 \
        --eval-interval "${eval_interval}" --eval-batches "${eval_batches}" \
        --log-interval "${log_interval}" \
        "${eval_args[@]}" "${save_args[@]}" \
        --results-base-folder "${RESULTS_DIR}" \
        --wandb-group "${group}" \
        --metrics-jsonl "${exp_dir}/metrics.jsonl"

    if [[ "${MODE}" == "full" && "${SAVE_CHECKPOINTS}" == "1" ]]; then
        remove_checkpoint_tree "${exp_dir}/ckpts"
    fi
    touch "${marker}"
    echo "RUN_COMPLETE queue=${QUEUE_ID} rho=${rho} experiment=${experiment}"
}

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for rho in "${RHOS[@]}"; do
    run_rho "${rho}"
done

echo "QUEUE_COMPLETE=${QUEUE_ID} MODE=${MODE}"
