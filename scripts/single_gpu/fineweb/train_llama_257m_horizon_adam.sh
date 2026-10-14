#!/usr/bin/env bash
set -euo pipefail

# Same Llama-257M/FineWeb recipe and global batch as the AdamW baseline.
# Examples:
#   DATASETS_DIR=/data/fineweb bash scripts/single_gpu/fineweb/train_llama_257m_horizon_adam.sh 1
#   HORIZON_BITS=12 HORIZON=64 LR=3e-4 bash scripts/single_gpu/fineweb/train_llama_257m_horizon_adam.sh 2
#   DRY_RUN=1 bash scripts/single_gpu/fineweb/train_llama_257m_horizon_adam.sh
# Extra src/main.py flags (including --resume-from) can follow the GPU count.
# HorizonAdam already compresses state. Do not combine it with --fp8-optim.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
NGPUS=${1:-1}
if (( $# > 0 )); then shift; fi
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-fp8-pretrain}
WANDB_GROUP=${WANDB_GROUP:-horizon_adam_257m}
ENABLE_WANDB=${ENABLE_WANDB:-1}
TORCHRUN=${TORCHRUN:-torchrun}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
HORIZON=${HORIZON:-32}
HORIZON_BITS=${HORIZON_BITS:-8}
BLOCK_SIZE=${BLOCK_SIZE:-256}
CHUNK_SIZE=${CHUNK_SIZE:-65536}
PROBES=${PROBES:-64}
MIN_SIZE=${MIN_SIZE:-4096}
SEED=${SEED:-0}
ROUNDING_SEED=${ROUNDING_SEED:-0}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"horizon_adam_257m_b${HORIZON_BITS}_h${HORIZON}_lr${LR}_wd${WEIGHT_DECAY}_s${SEED}"}

WANDB_ARGS=()
if [[ "${ENABLE_WANDB}" == "1" ]]; then
    WANDB_ARGS=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}"
                --wandb-tags horizon_adam bf16 257m)
fi

# The repository's DDP backend divides batch-size * acc-steps across ranks.
# Keep this product at 128, independent of NGPUS.
COMMAND=("${TORCHRUN}" --standalone --nproc_per_node="${NGPUS}" src/main.py
    --distributed-backend nccl --experiment-name "${EXPERIMENT_NAME}"
    --seed "${SEED}" --data-seed "${DATA_SEED:-1337}"
    --dataset fineweb --datasets-dir "${DATASETS_DIR}" --eval-cache-dir "${EVAL_CACHE_DIR}"
    --streaming --workers 8 --sequence-length 1024
    --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256 --dtype bfloat16
    --opt horizon_adam --lr "${LR}" --weight-decay "${WEIGHT_DECAY}"
    --beta1 0.9 --beta2 0.99 --grad-clip 1.0
    --horizon-adam-horizon "${HORIZON}" --horizon-adam-bits "${HORIZON_BITS}"
    --horizon-adam-block-size "${BLOCK_SIZE}" --horizon-adam-chunk-size "${CHUNK_SIZE}"
    --horizon-adam-probes "${PROBES}" --horizon-adam-min-size "${MIN_SIZE}"
    --horizon-adam-seed "${ROUNDING_SEED}"
    --scheduler cos_warmup_zero --warmup-steps "${WARMUP:-1963}" --iterations "${ITERATIONS:-39250}"
    --batch-size "${BATCH_SIZE:-16}" --acc-steps "${ACC_STEPS:-8}"
    --eval-interval 500 --eval-batches 32
    --downstream-eval-enabled --downstream-eval-interval 2000 --downstream-task-group basic_v2
    --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103
    --log-interval 50 --latest-ckpt-interval "${LATEST_CKPT_INTERVAL:-5000}"
    --results-base-folder "${RESULTS_DIR}"
    "${WANDB_ARGS[@]}" "$@")

cd "${REPO_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
else
    exec "${COMMAND[@]}"
fi
