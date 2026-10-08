#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE=${MODE:-smoke}
PRECISION=${PRECISION:?Set bf16, fp8_act, or bf16_g16}
case "$PRECISION" in bf16|fp8_act|bf16_g16) ;; *) exit 2 ;; esac
case "$MODE" in smoke|full) ;; *) exit 2 ;; esac
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs3/dimativator/softmuon-500m-20261008}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
GROUP=softmuon_500m_1xC_20261008
RANK_ID=${OMPI_COMM_WORLD_RANK:-0}
MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
NAME=softmuon_500m_${PRECISION}_${MODE}
mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/wandb"
exec > >(tee -a "$RESULTS_DIR/logs/${NAME}_rank${RANK_ID}.log") 2>&1
echo "RUN_START=$(date --iso-8601=seconds) PRECISION=$PRECISION MODE=$MODE GLOBAL_BATCH=128"
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
export WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey WANDB_DIR="$RESULTS_DIR/wandb"
export WANDB_MODE=online
export TRITON_CACHE_DIR="/tmp/softmuon-${PRECISION}-${MODE}-rank${RANK_ID}-$$"
ARGS=(--distributed-backend nccl --seed 0 --data-seed 1337 --dataset fineweb
    --datasets-dir "$DATASETS_DIR" --fineweb-replay-world-size 1 --fineweb-replay-layout concat
    --sequence-length 1024 --streaming --workers 8 --model llama --n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256
    --dtype bfloat16 --dropout 0 --opt softmuon --lr 1e-3 --weight-decay 1e-4
    --beta1 0.9 --beta2 0.95 --eps 1e-10 --grad-clip 1
    --batch-size 16 --acc-steps 8 --eval-batch-size 32 --iterations 75457 --scheduler wsd
    --warmup-steps 7000 --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine
    --softmuon-sign-fraction 0.9 --softmuon-eps 1e-4 --softmuon-newton-iters 10
    --no-local-save --latest-ckpt-interval 0 --permanent-ckpt-interval 0
    --experiment-name "$NAME" --results-base-folder "$RESULTS_DIR" --wandb-group "$GROUP"
    --wandb --wandb-project fp8-pretrain --wandb-tags SoftMuon 500M 1xC "$PRECISION" fp32_states
    --metrics-jsonl "$RESULTS_DIR/$GROUP/$NAME/metrics.jsonl")
if [[ "$PRECISION" == fp8_act ]]; then
    ARGS+=(--fp8 --fp8-fabit E4M3 --fp8-fwbit E4M3 --fp8-babit E5M2 --fp8-bwbit E5M2 --fp8-group-size 16)
elif [[ "$PRECISION" == bf16_g16 ]]; then
    ARGS+=(--model-parameter-dtype bfloat16)
fi
if [[ "$MODE" == smoke ]]; then
    python scripts/cloud/test_softmuon.py
    ARGS+=(--early-stop-iteration 2 --eval-interval 2 --eval-batches 1 --log-interval 1)
else
    test ! -e "$RESULTS_DIR/$GROUP/$NAME/metrics.jsonl"
    ARGS+=(--eval-interval 500 --eval-batches 32 --log-interval 50)
fi
"${LAUNCH[@]}" src/main.py "${ARGS[@]}"
if (( RANK_ID == 0 )); then echo "SOFTMUON_COMPLETE mode=$MODE precision=$PRECISION"; fi
