#!/usr/bin/env bash
set -euo pipefail
MODE=${MODE:-smoke}
case "$MODE" in smoke|full) ;; *) exit 2 ;; esac
RESULTS_DIR=${RESULTS_DIR:?Set a fresh RESULTS_DIR}
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
GROUP=figure6a_correct_checkpoint_20261005
RANK_ID=${OMPI_COMM_WORLD_RANK:-0}
MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/wandb"
exec > >(tee -a "$RESULTS_DIR/logs/${MODE}_rank${RANK_ID}.log") 2>&1
echo "RUN_START=$(date --iso-8601=seconds) MODE=$MODE WARMUP=7000 GLOBAL_BATCH=128 RANK=$RANK_ID"
df -h "$RESULTS_DIR" /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
python - <<'PY'
import json, os, torch
from pathlib import Path
m = json.loads((Path(os.environ.get('DATASETS_DIR', '/workspace-SR006.nfs3/dimativator/fineweb-h200-packed')) / 'packed_metadata.json').read_text())
assert m['format'] == 'packed_fineweb_h200_v3'
assert m['world_size'] == 2 and m['batch_size'] == 16 and m['iterations'] >= 75457
assert m['manifest_fingerprint'] == '7327154b810ec27cf5ca794aedcc3aea11796b218261ff24b5e2d3d2d283e00b'
assert torch.cuda.is_available()
print('DATA_ENV_OK', torch.__version__, flush=True)
PY
if (( MPI_SIZE > 1 )); then
    test "$MPI_SIZE" = 2
    export RANK=${RANK:-$RANK_ID} WORLD_SIZE=${WORLD_SIZE:-$MPI_SIZE}
    export LOCAL_RANK=${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
    export MASTER_ADDR=${MASTER_ADDR:-$(hostname -f)} MASTER_PORT=${MASTER_PORT:-29500}
    LAUNCH=(python)
else
    LAUNCH=(torchrun --standalone --nproc_per_node=2)
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True COAT_FP8_BACKEND=native
export WANDB_MODE=offline WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey WANDB_DIR="$RESULTS_DIR/wandb"
export TRITON_CACHE_DIR="/tmp/figure6a-${MODE}-rank${RANK_ID}-$$"
COMMON=(--distributed-backend nccl --seed 0 --data-seed 1337
    --dataset fineweb --datasets-dir "$DATASETS_DIR" --sequence-length 1024 --streaming --workers 8
    --model llama --n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256 --dtype bfloat16 --dropout 0
    --opt triton_coat_adamw --lr 1e-3 --weight-decay 1e-4 --beta1 0.9 --beta2 0.99 --eps 1e-7 --grad-clip 1
    --fp8-optim --fp8-qgroup-size 128 --fp8-first-order-bit E4M3 --fp8-second-order-bit E4M3 --fp8-expansion expand
    --scheduler wsd --warmup-steps 7000 --iterations 75457 --wsd-fract-decay 0.1 --wsd-final-lr-scale 0 --decay-type cosine
    --batch-size 16 --acc-steps 8 --eval-batch-size 32 --results-base-folder "$RESULTS_DIR" --wandb-group "$GROUP")
if [[ "$MODE" == smoke ]]; then
    test ! -e "$RESULTS_DIR/$GROUP/smoke_continuous"
    "${LAUNCH[@]}" src/main.py "${COMMON[@]}" --experiment-name smoke_continuous \
        --early-stop-iteration 2 --eval-interval 2 --eval-batches 1 --log-interval 1 \
        --inter-ckpts 1 2 --latest-ckpt-interval 0
    "${LAUNCH[@]}" src/main.py "${COMMON[@]}" --experiment-name smoke_resume \
        --resume-from "$RESULTS_DIR/$GROUP/smoke_continuous/ckpts/1" \
        --early-stop-iteration 2 --eval-interval 2 --eval-batches 1 --log-interval 1 \
        --inter-ckpts 2 --latest-ckpt-interval 0
    if (( RANK_ID == 0 )); then
        python scripts/cloud/audit_figure6a_checkpoint.py "$RESULTS_DIR/$GROUP/smoke_continuous/ckpts/2" \
            --iteration 2 --compare "$RESULTS_DIR/$GROUP/smoke_resume/ckpts/2"
        echo FIGURE6A_CHECKPOINT_SMOKE_COMPLETE
    fi
else
    NAME=llama500M_adamw_fp8_states_1xC_2gpu_global128
    test ! -e "$RESULTS_DIR/$GROUP/$NAME"
    free_bytes=$(df -B1 --output=avail "$RESULTS_DIR" | tail -n 1 | tr -d ' ')
    (( free_bytes > 15000000000 )) || { echo INSUFFICIENT_CHECKPOINT_DISK; exit 3; }
    "${LAUNCH[@]}" src/main.py "${COMMON[@]}" --experiment-name "$NAME" \
        --eval-interval 500 --eval-batches 32 --log-interval 50 \
        --inter-ckpts 67911 75457 --latest-ckpt-interval 10000 \
        --wandb --wandb-project fp8-pretrain --metrics-jsonl "$RESULTS_DIR/$GROUP/$NAME/metrics.jsonl"
    if (( RANK_ID == 0 )); then
        for step in 67911 75457; do
            python scripts/cloud/audit_figure6a_checkpoint.py "$RESULTS_DIR/$GROUP/$NAME/ckpts/$step" --iteration "$step"
        done
        echo FIGURE6A_CORRECT_CHECKPOINT_COMPLETE iter=75457
    fi
fi
