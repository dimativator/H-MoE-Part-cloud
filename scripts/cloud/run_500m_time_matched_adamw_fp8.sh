#!/usr/bin/env bash
set -euo pipefail

# Resume the verified pre-decay FP8-state model on a single GPU. The published
# worker_0.pt cursor is 5,000 iterations stale, so construct a separate packed
# replay cursor from the checkpoint's exact iteration. Do not alter HF files.
MODE=${MODE:-smoke}
case "${MODE}" in smoke|full) ;; *) echo "Unsupported MODE=${MODE}" >&2; exit 2 ;; esac
SOURCE_DIR=${SOURCE_DIR:-/workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/adamw-source}
SOURCE_FILES=${SOURCE_DIR}/adamw_fp8_states/pre_decay
REPAIRED_DIR=${SOURCE_DIR}/packed-replay-at-67911
DATASETS_DIR=${DATASETS_DIR:-/workspace-SR006.nfs3/dimativator/fineweb-h200-packed}
RESULTS_DIR=${RESULTS_DIR:-/workspace-SR006.nfs2/dimativator/500m-time-matched-20260925/adamw}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-/home/jovyan/evals_cache}
WANDB_GROUP=500M_time_matched_muon_fp8_1xC_20260925
EXPERIMENT_NAME=llama500M_adamw_fp8_time_matched_1gpu
METRICS_JSONL=${RESULTS_DIR}/${WANDB_GROUP}/${EXPERIMENT_NAME}/metrics.jsonl
ITERATIONS=92807

test -f "${SOURCE_FILES}/main.pt"
test -f "${SOURCE_FILES}/worker_0.pt"
test -f "${DATASETS_DIR}/packed_metadata.json"
mkdir -p "${RESULTS_DIR}/logs" "${RESULTS_DIR}/wandb" "${EVAL_CACHE_DIR}" "${REPAIRED_DIR}"
LOG_FILE=${RESULTS_DIR}/logs/${EXPERIMENT_NAME}_${MODE}.log
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "RUN_START=$(date --iso-8601=seconds) MODE=${MODE} ITERATIONS=${ITERATIONS} DECAY_STEPS=7546"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

SOURCE_FILES="${SOURCE_FILES}" REPAIRED_DIR="${REPAIRED_DIR}" DATASETS_DIR="${DATASETS_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

import torch

source = Path(os.environ['SOURCE_FILES'])
repaired = Path(os.environ['REPAIRED_DIR'])
metadata = json.loads((Path(os.environ['DATASETS_DIR']) / 'packed_metadata.json').read_text())
main = torch.load(source / 'main.pt', map_location='cpu', mmap=True, weights_only=False)
worker = torch.load(source / 'worker_0.pt', map_location='cpu', weights_only=False)
old_reader = worker['train_reader_state']
iteration = int(main['itr'])
microstep = 4 * iteration
assert iteration == 67911
assert int(old_reader['step']) == microstep - 20000
assert old_reader['reader_type'] == 'fineweb_train_reader_v1'
assert int(old_reader['batch_size']) == 16
assert int(old_reader['sequence_length']) == 1024
stream = old_reader['stream_state']
assert int(stream['rank']) == 0 and int(stream['world_size']) == 2
assert stream['manifest_fingerprint'] == metadata['manifest_fingerprint']
assert stream['split_plan_fingerprint'] == metadata['split_plan_fingerprint']
assert metadata['format'] == 'packed_fineweb_h200_v3'
assert int(metadata['iterations']) >= 92807
assert int(metadata['world_size']) == 2 and int(metadata['batch_size']) == 16
assert int(main['scheduler']['last_epoch']) == iteration
first_state = next(iter(main['optimizer']['state'].values()))
assert 'scale_exp_avg' in first_state and 'scale_exp_avg_sq' in first_state

def source_state(rank):
    return {
        'reader_type': 'packed_fineweb_train_reader_v1',
        'rank': rank,
        'batch_size': 16,
        'sequence_length': 1024,
        'step': microstep,
    }

worker['train_reader_state'] = {
    'reader_type': 'fineweb_replay_train_reader_v1',
    'batch_size': 32,
    'sequence_length': 1024,
    'source_world_size': 2,
    'step': microstep,
    'source_states': [source_state(0), source_state(1)],
}
repaired_main = repaired / 'main.pt'
if not repaired_main.exists():
    repaired_main.symlink_to(source / 'main.pt')
elif repaired_main.resolve() != (source / 'main.pt').resolve():
    raise RuntimeError('Repaired main.pt points to an unexpected source')
torch.save(worker, repaired / 'worker_0.pt')
print(f'ADAMW_SOURCE_ITER={iteration} ORIGINAL_WORKER_STEP={old_reader["step"]}', flush=True)
print(f'REPAIRED_PACKED_REPLAY_STEP={microstep} DATA_ITERATIONS={metadata["iterations"]}', flush=True)
print('REPAIRED_SOURCE_AUDIT_OK', flush=True)
PY

PYTHON_BIN=$(command -v python)
"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.__version__.startswith('2.9.1'), torch.__version__
assert torch.cuda.is_available()
print('ENVIRONMENT_CHECK=ok', torch.__version__, flush=True)
PY
WSD_FRACT_DECAY=$("${PYTHON_BIN}" - <<'PY'
fraction = 7546.5 / 92807
assert int(92807 * fraction) == 7546
print(repr(fraction))
PY
)

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True FINEWEB_LOG_DATA_HASHES=1
export WANDB_BASE_URL=https://wandb-radfan.ru WANDB_ENTITY=andrey
export WANDB_MODE=offline WANDB_DIR="${RESULTS_DIR}/wandb"
export TRITON_CACHE_DIR="/tmp/triton-500m-time-matched-adamw-${MODE}-$$"
mkdir -p "${TRITON_CACHE_DIR}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

EXTRA_ARGS=()
if [[ "${MODE}" == "smoke" ]]; then
    EXTRA_ARGS=(--early-stop-iteration 67913)
    EXPERIMENT_NAME=${EXPERIMENT_NAME}_smoke
else
    EXTRA_ARGS=(
        --downstream-eval-enabled --downstream-eval-interval 2000
        --downstream-task-group basic_v2
        --lm-eval-enabled --lm-eval-interval 2000 --lm-eval-datasets wikitext103
        --wandb --wandb-project fp8-pretrain --wandb-group "${WANDB_GROUP}"
        --wandb-tags fineweb fp8_optimizer_states 500M 1xC time_matched adamw
        --metrics-jsonl "${METRICS_JSONL}"
    )
fi

"$(command -v torchrun)" --standalone --nproc_per_node=1 src/main.py \
    --distributed-backend nccl \
    --experiment-name "${EXPERIMENT_NAME}" \
    --seed 0 --data-seed 1337 \
    --dataset fineweb --datasets-dir "${DATASETS_DIR}" \
    --fineweb-replay-world-size 2 --fineweb-replay-layout concat \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length 1024 --streaming --workers 8 \
    --model llama --n-layer 18 --n-embd 1280 --n-head 20 --multiple-of 256 \
    --dtype bfloat16 \
    --opt triton_coat_adamw --lr 1e-3 --weight-decay 1e-4 \
    --beta1 0.9 --beta2 0.99 --grad-clip 1.0 \
    --fp8-optim --fp8-qgroup-size 128 \
    --fp8-first-order-bit E4M3 --fp8-second-order-bit E4M3 --fp8-expansion expand \
    --scheduler wsd --warmup-steps 2000 --iterations "${ITERATIONS}" \
    --wsd-fract-decay "${WSD_FRACT_DECAY}" --wsd-final-lr-scale 0 --decay-type cosine \
    --batch-size 32 --acc-steps 4 --eval-batch-size 32 \
    --eval-interval 500 --eval-batches 32 --log-interval 50 \
    --resume-from "${REPAIRED_DIR}" --no-local-save \
    --results-base-folder "${RESULTS_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "TIME_MATCHED_ADAMW_${MODE^^}_COMPLETE iter=${ITERATIONS}"
