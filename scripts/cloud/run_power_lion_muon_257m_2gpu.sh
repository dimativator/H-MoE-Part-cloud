#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE=${MODE:-campaign}
LR=${LR:?Set one of the four approved learning rates}
case "$LR" in 2e-3|1e-3|5e-4|1e-4) ;; *) exit 2 ;; esac
case "$MODE" in
    campaign)
        MODE=smoke bash "$0"
        MODE=full bash "$0"
        exit 0 ;;
    smoke|full) ;;
    *) exit 2 ;;
esac
RESULTS_DIR=${RESULTS_DIR:-/home/jovyan/dimativator/power-lion-muon-257m-lr-20261020}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
GROUP=power_lion_muon_257m_lr_20261020
RANK_ID=${OMPI_COMM_WORLD_RANK:-0}
MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
NAME=power_lion_muon_257m_lr${LR}_wd1e-1_${MODE}
mkdir -p "$RESULTS_DIR/logs"
exec > >(tee -a "$RESULTS_DIR/logs/${NAME}_rank${RANK_ID}.log") 2>&1
echo "POWER_RUN_START=$(date --iso-8601=seconds) LR=$LR MODE=$MODE GLOBAL_BATCH=128"
df -h "$RESULTS_DIR" "$DATASETS_DIR"
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
export WANDB_MODE=disabled
echo "WANDB_DISABLED local_metrics_only=1 CHECKPOINTS_DISABLED=all"
if [[ "$MODE" == smoke ]]; then
    "${LAUNCH[@]}" scripts/cloud/test_power_lion_muon_gpu.py
fi
test ! -e "$RESULTS_DIR/$GROUP/$NAME/metrics.jsonl"
ARGS=(--distributed-backend nccl --seed 0 --data-seed 1337 --dataset fineweb
    --datasets-dir "$DATASETS_DIR" --fineweb-replay-world-size 1 --fineweb-replay-layout concat
    --sequence-length 1024 --streaming --workers 8 --model llama --n-layer 12 --n-embd 1024 --n-head 8 --multiple-of 256
    --dtype bfloat16 --dropout 0 --opt power_lion_muon --lr "$LR" --weight-decay 1e-1
    --beta1 0.9 --beta2 0.99 --eps 1e-8 --grad-clip 1
    --batch-size 16 --acc-steps 8 --eval-batch-size 32 --iterations 39250 --scheduler wsd
    --warmup-steps 2000 --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine
    --power-lion-muon-probability 0.2 --power-lion-muon-seed 20261020 --power-lion-muon-momentum 0.95
    --power-lion-muon-start-fraction 0.9 --power-lion-muon-exponent-start 0 --power-lion-muon-exponent-end 1
    --power-lion-muon-exponent-gamma 1 --power-lion-muon-ns-steps 30 --power-lion-muon-polynomial-terms 12
    --no-local-save --latest-ckpt-interval 0 --permanent-ckpt-interval 0
    --experiment-name "$NAME" --results-base-folder "$RESULTS_DIR" --wandb-group "$GROUP"
    --metrics-jsonl "$RESULTS_DIR/$GROUP/$NAME/metrics.jsonl")
if [[ "$MODE" == smoke ]]; then
    ARGS+=(--early-stop-iteration 10 --eval-interval 10 --eval-batches 1 --log-interval 1)
else
    ARGS+=(--eval-interval 500 --eval-batches 32 --log-interval 50)
fi
"${LAUNCH[@]}" src/main.py "${ARGS[@]}"
echo "POWER_LION_MUON_COMPLETE mode=$MODE lr=$LR rank=$RANK_ID"
