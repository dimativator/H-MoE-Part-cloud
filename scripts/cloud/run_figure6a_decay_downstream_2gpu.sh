#!/usr/bin/env bash
set -euo pipefail
RESULTS_DIR=${RESULTS_DIR:-/home/jovyan/dimativator/figure6a-fix-decay-downstream-20261013}
DATASETS_DIR=/workspace-SR006.nfs3/dimativator/fineweb-h200-packed
SOURCE_CKPT=/workspace-SR006.nfs3/dimativator/figure6a-adamw-fp8-retrain-20261005/figure6a_correct_checkpoint_20261005/llama500M_adamw_fp8_states_1xC_2gpu_global128/ckpts/67911
GROUP=figure6a_fix_decay_downstream_20261013
NAME=adamw_fp8_states_fix_decay_global128
RANK_ID=${OMPI_COMM_WORLD_RANK:-0}
MPI_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test ! -e "$RESULTS_DIR/$GROUP/$NAME"
mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/wandb"
exec > >(tee -a "$RESULTS_DIR/logs/decay_rank${RANK_ID}.log") 2>&1
echo "DECAY_START=$(date --iso-8601=seconds) SOURCE=$SOURCE_CKPT GLOBAL_BATCH=128 RANK=$RANK_ID"
df -h "$RESULTS_DIR" /workspace-SR006.nfs3
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
free_bytes=$(df -B1 --output=avail "$RESULTS_DIR" | tail -n 1 | tr -d ' ')
(( free_bytes > 6000000000 )) || { echo INSUFFICIENT_FINAL_CHECKPOINT_DISK; exit 3; }
if (( RANK_ID == 0 )); then
    python scripts/cloud/audit_figure6a_checkpoint.py "$SOURCE_CKPT" --iteration 67911
fi
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
export FIGURE6A_CHECKPOINT_AUDIT=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey WANDB_DIR="$RESULTS_DIR/wandb"
export TRITON_CACHE_DIR="/tmp/figure6a-decay-rank${RANK_ID}-$$"
DECAY_FRACTION=$(python -c 'print(repr(7546.5 / 75457))')
"${LAUNCH[@]}" src/main.py \
    --distributed-backend nccl --seed 0 --data-seed 1337 \
    --dataset fineweb --datasets-dir "$DATASETS_DIR" --sequence-length 1024 --streaming --workers 8 \
    --model llama --n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256 --dtype bfloat16 --dropout 0 \
    --opt triton_coat_adamw --lr 1e-3 --weight-decay 1e-4 --beta1 0.9 --beta2 0.99 --eps 1e-7 --grad-clip 1 \
    --fp8-optim --fp8-qgroup-size 128 --fp8-first-order-bit E4M3 --fp8-second-order-bit E4M3 --fp8-expansion expand \
    --scheduler wsd --warmup-steps 7000 --iterations 75457 --wsd-fract-decay "$DECAY_FRACTION" --wsd-final-lr-scale 0 --decay-type cosine \
    --batch-size 16 --acc-steps 8 --eval-batch-size 32 \
    --resume-from "$SOURCE_CKPT" --results-base-folder "$RESULTS_DIR" --wandb-group "$GROUP" --experiment-name "$NAME" \
    --eval-interval 500 --eval-batches 32 --log-interval 50 --inter-ckpts 75457 --latest-ckpt-interval 0 \
    --downstream-eval-enabled --downstream-eval-interval 75457 --downstream-task-group basic_v2 \
    --wandb --wandb-project fp8-pretrain --metrics-jsonl "$RESULTS_DIR/$GROUP/$NAME/metrics.jsonl"
if (( RANK_ID == 0 )); then
    python scripts/cloud/audit_figure6a_checkpoint.py "$RESULTS_DIR/$GROUP/$NAME/ckpts/75457" --iteration 75457
    python - "$RESULTS_DIR/$GROUP/$NAME/metrics.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert any(row.get('event') == 'validation' and row.get('iter') == 75457 for row in rows)
assert any(row.get('event') == 'downstream' and row.get('iter') == 75457 for row in rows)
print('FIGURE6A_DECAY_DOWNSTREAM_COMPLETE iter=75457', flush=True)
PY
fi
