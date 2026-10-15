#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE=${MODE:-smoke}
case "$MODE" in
    smoke|full) ;;
    campaign)
        # The same MPI allocation runs tests first, then all six full runs.
        # set -e prevents training from starting if any smoke command fails.
        MODE=smoke bash "$0"
        MODE=full bash "$0"
        exit 0
        ;;
    *) exit 2 ;;
esac
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/lionmuon-probability-257m-20261012}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
GROUP=lionmuon_probability_257m_1xC_20261012
RANK_ID=${OMPI_COMM_WORLD_RANK:-0}
MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
mkdir -p "$RESULTS_DIR/logs"
exec > >(tee -a "$RESULTS_DIR/logs/queue_${MODE}_rank${RANK_ID}.log") 2>&1
echo "QUEUE_START=$(date --iso-8601=seconds) MODE=$MODE GLOBAL_BATCH=128"
df -h "$RESULTS_DIR"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
if (( MPI_SIZE > 1 )); then
    test "$MPI_SIZE" = 2
    export RANK=${RANK:-$RANK_ID} WORLD_SIZE=2 LOCAL_RANK=${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)} MASTER_PORT=${MASTER_PORT:-29500}
    LAUNCH=(python)
else
    LAUNCH=(torchrun --standalone --nproc_per_node=2)
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
WANDB_ARGS=()
if [[ "${ENABLE_WANDB:-1}" == 1 ]]; then
    mkdir -p "$RESULTS_DIR/wandb"
    export WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey WANDB_DIR="$RESULTS_DIR/wandb" WANDB_MODE=online
    WANDB_ARGS=(--wandb --wandb-project fp8-pretrain)
else
    export WANDB_MODE=disabled
    echo "WANDB_DISABLED local_metrics_only=1"
fi
if [[ "$MODE" == smoke ]]; then
    "${LAUNCH[@]}" scripts/cloud/test_lion_muon_probability.py
fi
for schedule in constant linear_up linear_down decay_up decay_down periodic; do
    NAME=lionmuon_257m_${schedule}_${MODE}
    test ! -e "$RESULTS_DIR/$GROUP/$NAME/metrics.jsonl"
    ARGS=(--distributed-backend nccl --seed 0 --data-seed 1337 --dataset fineweb
        --datasets-dir "$DATASETS_DIR" --fineweb-replay-world-size 1 --fineweb-replay-layout concat
        --sequence-length 1024 --streaming --workers 8 --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256
        --dtype bfloat16 --dropout 0 --opt lion_muon_probability --lr 1e-3 --weight-decay 1e-4
        --beta1 0.9 --beta2 0.99 --eps 1e-8 --grad-clip 1
        --batch-size 16 --acc-steps 8 --eval-batch-size 32 --iterations 39250 --scheduler wsd
        --warmup-steps 2000 --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine
        --lion-muon-probability-schedule "$schedule" --lion-muon-probability-seed 20261012
        --no-local-save --latest-ckpt-interval 0 --permanent-ckpt-interval 0
        --experiment-name "$NAME" --results-base-folder "$RESULTS_DIR" --wandb-group "$GROUP"
        "${WANDB_ARGS[@]}"
        --metrics-jsonl "$RESULTS_DIR/$GROUP/$NAME/metrics.jsonl")
    if [[ "$MODE" == smoke ]]; then
        ARGS+=(--early-stop-iteration 2 --eval-interval 2 --eval-batches 1 --log-interval 1)
    else
        ARGS+=(--eval-interval 500 --eval-batches 32 --log-interval 50)
    fi
    echo "LION_MUON_RUN_START schedule=$schedule mode=$MODE rank=$RANK_ID"
    "${LAUNCH[@]}" src/main.py "${ARGS[@]}" > >(tee -a "$RESULTS_DIR/logs/${NAME}_rank${RANK_ID}.log") 2>&1
    echo "LION_MUON_COMPLETE schedule=$schedule mode=$MODE rank=$RANK_ID"
done
echo "LION_MUON_QUEUE_COMPLETE mode=$MODE rank=$RANK_ID"
